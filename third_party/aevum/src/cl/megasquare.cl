// Megapass fused kernels: fftMiddleIn -> tailSquare -> fftMiddleOut in ONE launch per field.
//
// Design (verified adversarially, workflow wf_df0a628e-eae, v1 = full-size scratch buffers):
//  - 6144 WGs x 128 threads. Every WG's work is looked up in a host-built schedule map
//    (bufMegaMap); the kernel NEVER derives gx/gy/line from get_group_id(0).
//  - Flat WG ids are cohort-major: cohort c = mirror-closed width-chunk pair {c, 31-c},
//    each cohort emits [MidIn block][Tail block][MidOut block] in ascending flat id.
//    The host asserts, at startup, that EVERY dependency edge has producer flat id <
//    consumer flat id (forward-only), which is the same scheduler premise carryFused's
//    production stairway relies on (WGs dispatched in nondecreasing id order).
//  - Cross-WG synchronization: epoch-stamped flags in bufMegaReady, using verbatim the
//    carryFused OLD_FENCE publish/consume recipe (carryfused.cl ~1887-1918):
//      publish: payload stores; write_mem_fence(GLOBAL); bar(); thread-0 atomic_store(flag, epoch)
//      consume: per-thread relaxed atomic_load spin until flag==epoch; bar(); read_mem_fence(GLOBAL)
//    Flags are never reset in-kernel (multi-consumer); the host passes a fresh epoch per
//    launch and re-zeroes the flag buffer before the u32 epoch wraps.
//  - Intermediates go to dedicated per-field scratch buffers scrA (MidIn->Tail) and
//    scrB (Tail->MidOut) in the exact stock layouts; scrA/scrB base pointers are already
//    per-field, so DISTGF31/DISTGF61 are NOT applied to them.
//
// Map layout (u32 entries, shared by both fields -- pure geometry):
//   map[0..6143]  = desc per flat WG id:
//       bits 30..31 = role: 0=MidIn, 1=Tail, 2=MidOut
//       MidIn : bits 0..4 = gx, bits 5..10 = gy
//       Tail  : bits 0..10 = g (line_u), bits 11..15 = gx_u, bits 16..20 = gx_v
//       MidOut: bits 0..4 = gxo, bits 5..10 = gyo, bits 11..21 = flag-table slot
//   map[6144 + slot*64 + lane], lane 0..63 = region-relative ready index of the tail flag
//       that MidOut thread 'lane' must spin on (host-generated = host-verified).
//
// Ready-flag regions in bufMegaReady (12288 u32, host-zeroed at alloc):
//   GF31: [0, 6144), GF61: [6144, 12288).
//   Within a region: midinReady = base + gx*64 + gy   (indices [0, 2048))
//                    tailReady  = base + 2048 + g     (indices [2048, 4096))
//                    [4096, 6144) reserved for the deferred v2 ring.
//
// MEGA_DEBUG=1 turns every unbounded spin into a bounded spin (~MEGA_SPIN_BOUND iters)
// with a diagnostic record appended to bufMegaDebug on expiry, so a schedule bug becomes
// a diagnosable wrong-result instead of a GPU hang. It also enables megaCanary, the
// ascending-dispatch canary kernel (mandatory startup gate on new arch/driver).

#include "base.cl"
#include "fft-middle.cl"
#include "tailsquare.cl"

#if MEGAPASS31 || MEGAPASS61

// ---------------------------------------------------------------------------
// Compile guards: the schedule map and the flag index math are built for the
// default geometry of shape 1:512:8:512. Any deviation invalidates them.
// ---------------------------------------------------------------------------
#if WIDTH != 512 || MIDDLE != 8 || SMALL_HEIGHT != 512
#error MEGAPASS requires WIDTH=512 MIDDLE=8 SMALL_HEIGHT=512
#endif
#if NW != 8 || NH != 8
#error MEGAPASS requires NW=8 and NH=8 (so G_W=G_H=64)
#endif
#if G_H != 64
#error MEGAPASS requires G_H == 64
#endif
#if TAIL_KERNELS != 2
#error MEGAPASS requires TAIL_KERNELS=2 (double-wide single-kernel tailSquare)
#endif
#if IN_WG != 128 || OUT_WG != 128 || IN_SIZEX != 16 || OUT_SIZEX != 16
#error MEGAPASS requires IN_WG=OUT_WG=128 and IN_SIZEX=OUT_SIZEX=16
#endif
#if PAD != 0
#error MEGAPASS requires PAD=0 (no-padding scratch layouts)
#endif
#if INPLACE
#error MEGAPASS is incompatible with INPLACE
#endif
#if LIFO_MID
#error MEGAPASS is incompatible with LIFO_MID
#endif
#if !MIDDLE_IN_LDS_TRANSPOSE || !MIDDLE_OUT_LDS_TRANSPOSE
#error MEGAPASS requires MIDDLE_IN_LDS_TRANSPOSE=1 and MIDDLE_OUT_LDS_TRANSPOSE=1 (contiguous out+=me block writes)
#endif
#if NONTEMPORAL
#error MEGAPASS requires NONTEMPORAL=0 (audited publish path)
#endif
#if MULTI_Q
#error MEGAPASS requires MULTI_Q=0 (single in-order queue)
#endif

// Number of WGs in the fused launch: 3 roles x 2048 WGs.
#define MEGA_NWGS       6144
// Region-relative flag index bases.
#define MEGA_TAILFLAG   2048
// bufMegaReady region base per field.
#define MEGA_RDY31      0
#define MEGA_RDY61      6144
// MidOut flag tables start here inside bufMegaMap.
#define MEGA_FLAGTAB    6144

// Diagnostic knob (-use MEGA_SPIN_NS=<n>): nanoseconds between ready-flag polls on the
// sm_70+ path. Default 256 (the measured-best production value; the default expansion is
// byte-identical to the previous hardcoded "nanosleep.u32 256"). 0 = empty spin.
#if !defined(MEGA_SPIN_NS)
#define MEGA_SPIN_NS 256
#endif

#if !defined(MEGA_SPIN_BOUND)
#if HAS_PTX >= 700 && MEGA_SPIN_NS > 0
#define MEGA_SPIN_BOUND 4000000u     // polls are ~MEGA_SPIN_NS(=256 default) apart -> bound ~1s
#else
#define MEGA_SPIN_BOUND 100000000u
#endif
#endif

#define MEGA_STR_(x) #x
#define MEGA_STR(x) MEGA_STR_(x)

// carryFused's spin() (carryfused.cl:9-18) is an empty loop on Nvidia. Here many threads
// poll concurrently, so throttle the L2 atomic-poll rate with nanosleep where available
// (sm_70+); ~256ns measured best on the 3090 (1024ns was slower, 0ns/empty was slower).
// The nanosleep is only emitted when the sm_70+ path exists AND MEGA_SPIN_NS > 0.
void megaSpin() {
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_s_sleep)
  __builtin_amdgcn_s_sleep(0);
#elif HAS_ASM
  __asm("s_sleep 0");
#elif HAS_PTX >= 700 && MEGA_SPIN_NS > 0
  __asm volatile("nanosleep.u32 " MEGA_STR(MEGA_SPIN_NS) ";");
#else
  // nothing: just spin
#endif
}

#if MEGA_DEBUG
// Diagnostic buffer layout: dbg[0] = number of timeouts (atomic counter);
// record i (i < 63) at dbg[8 + i*4] = {flatId, role, flagIdx, lastSeenValue}.
void megaRecordTimeout(P(u32) dbg, u32 flatId, u32 role, u32 flagIdx, u32 seen) {
  u32 slot = atomic_fetch_add((atomic_uint *) &dbg[0], 1u);
  if (slot < 63) {
    u32 base = 8 + slot * 4;
    dbg[base + 0] = flatId;
    dbg[base + 1] = role;
    dbg[base + 2] = flagIdx;
    dbg[base + 3] = seen;
  }
}
#endif

// Spin until ready[flagIdx] == epoch (relaxed load, exactly like carryfused.cl:1915).
// Under MEGA_DEBUG the spin is bounded; on expiry a diagnostic is recorded and *fail set.
// Returns 0 on success, 1 on (debug) timeout.
// MEGA_NOSYNC=1 (-use, diagnostic only): the wait becomes a no-op so the kernel's compute
// cost can be timed without any spin/sync overhead. Publishes stay (cheap stores).
// RESULTS ARE INVALID in this mode -- the host logs this loudly and skips GEC verdicts.
u32 megaSpinWait(P(u32) ready, u32 flagIdx, u32 epoch, u32 role, P(u32) dbg, local u32 *fail) {
#if MEGA_NOSYNC
  (void) ready; (void) flagIdx; (void) epoch; (void) role; (void) dbg; (void) fail;
  return 0;
#elif MEGA_DEBUG
  u32 n = 0;
  u32 seen;
  while ((seen = atomic_load_explicit((atomic_uint *) &ready[flagIdx], memory_order_relaxed, memory_scope_device)) != epoch) {
    megaSpin();
    if (++n > MEGA_SPIN_BOUND) {
      megaRecordTimeout(dbg, (u32) get_group_id(0), role, flagIdx, seen);
      *fail = 1;
      return 1;
    }
  }
#else
  while (atomic_load_explicit((atomic_uint *) &ready[flagIdx], memory_order_relaxed, memory_scope_device) != epoch) { megaSpin(); }
#endif
  return 0;
}

// (A single-spinner sequential variant of the waits was measured on the 3090 and was
// clearly slower -- 64 dependent L2 round-trips serialize the observation; the parallel
// one-flag-per-thread spin below completes one round-trip after the last producer.)

// Kernels declare 'local u32 megaFail' only under MEGA_DEBUG; this macro supplies the
// matching argument (never dereferenced when !MEGA_DEBUG).
#if MEGA_DEBUG
#define MEGA_FAIL_PTR (&megaFail)
#else
#define MEGA_FAIL_PTR ((local u32 *) 0)
#endif

// Publish one flag on behalf of the whole WG, verbatim the carryFused OLD_FENCE recipe
// (carryfused.cl:1887-1891): all payload stores done -> write fence -> bar -> thread-0 store.
void megaPublish(P(u32) ready, u32 flagIdx, u32 epoch) {
  write_mem_fence(CLK_GLOBAL_MEM_FENCE);
  bar();
  if (get_local_id(0) == 0) { atomic_store((atomic_uint *) &ready[flagIdx], epoch); }
}

#if MEGA_DEBUG
// Ascending-dispatch canary: WG g waits (bounded) for WG g-1's flag, then publishes its
// own. If workgroups were dispatched out of id order without preemption-friendliness,
// timeouts get recorded. Uses flags [0, MEGA_NWGS) of bufMegaReady (epoch-stamped, so
// later megaSquare launches are unaffected by the stale values).
KERNEL(128) megaCanary(P(u32) ready, u32 epoch, P(u32) dbg) {
  u32 g = get_group_id(0);
  u32 me = get_local_id(0);
  if (g > 0 && me == 0) {
    u32 n = 0;
    u32 seen;
    while ((seen = atomic_load_explicit((atomic_uint *) &ready[g - 1], memory_order_relaxed, memory_scope_device)) != epoch) {
      megaSpin();
      if (++n > MEGA_SPIN_BOUND) {
        megaRecordTimeout(dbg, g, 3, g - 1, seen);
        break;                    // still publish below so the grid drains in bounded time
      }
    }
  }
  bar();
  write_mem_fence(CLK_GLOBAL_MEM_FENCE);
  if (me == 0) { atomic_store((atomic_uint *) &ready[g], epoch); }
}
#endif

// ---------------------------------------------------------------------------
// megaSquareGF31
// ---------------------------------------------------------------------------
#if NTT_GF31 && MEGAPASS31

KERNEL(128) megaSquareGF31(P(T2) out, CP(T2) in, u32 epoch, P(T2) scrA, P(T2) scrB,
                           P(u32) ready, CP(u32) map, Trig trigM, Trig trigH, P(u32) dbg) {
  // LDS union: tail needs 2*LDS_BYTES = 9088 bytes; MidIn/MidOut need Z31[1024] = 4096 bytes.
  local GF31 megaLds[2 * LDS_BYTES / sizeof(GF31)];

  u32 me = get_local_id(0);
  u32 flatId = get_group_id(0);
  u32 desc = map[flatId];
  u32 role = desc >> 30;

  CP(GF31) in31  = (CP(GF31)) (in  + DISTGF31);   // stock buf3 field region (post-carryFused)
  P(GF31)  out31 = (P(GF31))  (out + DISTGF31);   // stock buf field region (pre-carryFused)
  P(GF31)  scrA31 = (P(GF31)) scrA;               // dedicated per-field scratch: NO DIST offset
  P(GF31)  scrB31 = (P(GF31)) scrB;
  TrigGF31 trigM31 = (TrigGF31) (trigM + DISTMTRIGGF31);
  TrigGF31 trigH31 = (TrigGF31) (trigH + DISTHTRIGGF31);
  P(u32) rdy = ready + MEGA_RDY31;

#if MEGA_DEBUG
  local u32 megaFail;
  if (me == 0) { megaFail = 0; }
  bar();
#endif

  if (role == 0) {
    // ------------------------- MidIn: fftmiddlein.cl fftMiddleInGF31 body -------------------------
    u32 gx = desc & 31;
    u32 gy = (desc >> 5) & 63;

    GF31 u[MIDDLE];
    u32 SIZEY = IN_WG / IN_SIZEX;
    u32 mx = me % IN_SIZEX;
    u32 my = me / IN_SIZEX;
    u32 x = gx * IN_SIZEX + mx;
    u32 y = gy * SIZEY + my;

    readMiddleInLine(u, in31, y, x);
    middleMul2(u, x, y, trigM31);
    fft_MIDDLE(u);
    middleMul(u, y, trigM31);

    local Z31 *lds = (local Z31 *) megaLds;
    middleShuffle(lds, u, IN_WG, IN_SIZEX);
    writeMiddleInLine(scrA31 + me, u, gy, gx);

    megaPublish(rdy, gx * 64 + gy, epoch);

  } else if (role == 1) {
    // ------------------------- Tail: tailsquare.cl tailSquareGF31 body -------------------------
    u32 g    = desc & 2047;
    u32 gx_u = (desc >> 11) & 31;
    u32 gx_v = (desc >> 16) & 31;

    // Consume: half-WG threads spin the 64 gy-chunk flags of column chunk gx_u, the other
    // half those of gx_v (host-packed bases; the flag SET is exactly what
    // validateMegaSchedule checked; gx_u == gx_v duplicates are harmless).
    megaSpinWait(rdy, ((me < G_H) ? gx_u : gx_v) * 64 + (me % G_H), epoch, 1, dbg, MEGA_FAIL_PTR);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
#if MEGA_DEBUG
    if (megaFail) { return; }
#endif

    GF31 u[NH];
    u32 H = ND / SMALL_HEIGHT;
    u32 line_u = g;
    u32 line_v = line_u ? H - line_u : (H / 2);
    u32 lowMe = me % G_H;
    bool isSecondHalf = me >= G_H;
    u32 line = !isSecondHalf ? line_u : line_v;

    readTailFusedLine(scrA31, u, line, lowMe);

    // zerohack is 0 for every group id below 131072 in the stock kernel; flat ids < 6144.
    fft_HEIGHT1(megaLds, u, trigH31, 2, lowMe);

#if TAIL_TRIGS31 >= 1
    u32 height_trigs = SMALL_HEIGHT * 1;
    GF31 trig = TFLOAD(&trigH31[height_trigs + lowMe]);
    GF31 mult = TSLOAD(&trigH31[height_trigs + G_H + line_u * 2 + isSecondHalf]);
    trig = cmul(trig, mult);
#else
    u32 height_trigs = SMALL_HEIGHT * 1;
    GF31 trig = TOLOAD(&trigH31[height_trigs + line_u * G_H * 2 + me]);
#endif

    if (line_u == 0) {
      reverse2(megaLds, u);
      pairSq2_special(u, trig);
      reverse2(megaLds, u);
    } else {
      revCrossLine(megaLds, u);
      pairSq(NH / 2, u, u + NH / 2, trig, false);
      revCrossLine(megaLds, u);
    }

    fft_HEIGHT2(megaLds, u, trigH31, 2, lowMe);
    writeTailFusedLine(u, scrB31, transPos(line, MIDDLE, WIDTH), lowMe);

    megaPublish(rdy, MEGA_TAILFLAG + g, epoch);

  } else {
    // ------------------------- MidOut: fftmiddleout.cl fftMiddleOutGF31 body -------------------------
    u32 gxo  = desc & 31;
    u32 gyo  = (desc >> 5) & 63;
    u32 slot = (desc >> 11) & 2047;

    // Consume: threads 0..63 each spin one host-verified tail flag from the map's flag
    // table (the 8 producers of each of the 8 memline columns this WG reads); threads
    // 64..127 have nothing to wait on and go straight to the barrier.
    if (me < 64) {
      megaSpinWait(rdy, map[MEGA_FLAGTAB + slot * 64 + me], epoch, 2, dbg, MEGA_FAIL_PTR);
    }
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
#if MEGA_DEBUG
    if (megaFail) { return; }
#endif

    GF31 u[MIDDLE];
    u32 SIZEY = OUT_WG / OUT_SIZEX;
    u32 mx = me % OUT_SIZEX;
    u32 my = me / OUT_SIZEX;
    u32 x = gxo * OUT_SIZEX + mx;
    u32 y = gyo * SIZEY + my;

    readMiddleOutLine(u, scrB31, y, x);
    middleMul(u, x, trigM31);
    fft_MIDDLE(u);
    middleMul2(u, y, x, trigM31);

    local Z31 *lds = (local Z31 *) megaLds;
    middleShuffle(lds, u, OUT_WG, OUT_SIZEX);
    writeMiddleOutLine(out31 + me, u, gyo, gxo);
    // No publish in v1: the following carryFused launch is a natural barrier.
  }
}

#endif // NTT_GF31 && MEGAPASS31

// ---------------------------------------------------------------------------
// megaSquareGF61
// ---------------------------------------------------------------------------
#if NTT_GF61 && MEGAPASS61

KERNEL(128) megaSquareGF61(P(T2) out, CP(T2) in, u32 epoch, P(T2) scrA, P(T2) scrB,
                           P(u32) ready, CP(u32) map, Trig trigM, Trig trigH, P(u32) dbg) {
  // LDS union: tail needs 2*LDS_BYTES = 9088 bytes; MidIn/MidOut need Z61[1024] = 8192 bytes.
  local GF61 megaLds[2 * LDS_BYTES / sizeof(GF61)];

  u32 me = get_local_id(0);
  u32 flatId = get_group_id(0);
  u32 desc = map[flatId];
  u32 role = desc >> 30;

  CP(GF61) in61  = (CP(GF61)) (in  + DISTGF61);   // stock buf3 field region (post-carryFused)
  P(GF61)  out61 = (P(GF61))  (out + DISTGF61);   // stock buf field region (pre-carryFused)
  P(GF61)  scrA61 = (P(GF61)) scrA;               // dedicated per-field scratch: NO DIST offset
  P(GF61)  scrB61 = (P(GF61)) scrB;
  TrigGF61 trigM61 = (TrigGF61) (trigM + DISTMTRIGGF61);
  TrigGF61 trigH61 = (TrigGF61) (trigH + DISTHTRIGGF61);
  P(u32) rdy = ready + MEGA_RDY61;

#if MEGA_DEBUG
  local u32 megaFail;
  if (me == 0) { megaFail = 0; }
  bar();
#endif

  if (role == 0) {
    // ------------------------- MidIn: fftmiddlein.cl fftMiddleInGF61 body -------------------------
    u32 gx = desc & 31;
    u32 gy = (desc >> 5) & 63;

    GF61 u[MIDDLE];
    u32 SIZEY = IN_WG / IN_SIZEX;
    u32 mx = me % IN_SIZEX;
    u32 my = me / IN_SIZEX;
    u32 x = gx * IN_SIZEX + mx;
    u32 y = gy * SIZEY + my;

    readMiddleInLine(u, in61, y, x);
    middleMul2(u, x, y, trigM61);
    fft_MIDDLE(u);
    middleMul(u, y, trigM61);

    local Z61 *lds = (local Z61 *) megaLds;
    middleShuffle(lds, u, IN_WG, IN_SIZEX);
    writeMiddleInLine(scrA61 + me, u, gy, gx);

    megaPublish(rdy, gx * 64 + gy, epoch);

  } else if (role == 1) {
    // ------------------------- Tail: tailsquare.cl tailSquareGF61 body -------------------------
    u32 g    = desc & 2047;
    u32 gx_u = (desc >> 11) & 31;
    u32 gx_v = (desc >> 16) & 31;

    // Half-WG spins gx_u's 64 gy-chunk flags, other half gx_v's (host-verified set).
    megaSpinWait(rdy, ((me < G_H) ? gx_u : gx_v) * 64 + (me % G_H), epoch, 1, dbg, MEGA_FAIL_PTR);
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
#if MEGA_DEBUG
    if (megaFail) { return; }
#endif

    GF61 u[NH];
    u32 H = ND / SMALL_HEIGHT;
    u32 line_u = g;
    u32 line_v = line_u ? H - line_u : (H / 2);
    u32 lowMe = me % G_H;
    bool isSecondHalf = me >= G_H;
    u32 line = !isSecondHalf ? line_u : line_v;

    readTailFusedLine(scrA61, u, line, lowMe);

    fft_HEIGHT1(megaLds, u, trigH61, 2, lowMe);

#if TAIL_TRIGS61 >= 1
    u32 height_trigs = SMALL_HEIGHT * 1;
    GF61 trig = TFLOAD(&trigH61[height_trigs + lowMe]);
    GF61 mult = TSLOAD(&trigH61[height_trigs + G_H + line_u * 2 + isSecondHalf]);
    trig = cmul(trig, mult);
#else
    u32 height_trigs = SMALL_HEIGHT * 1;
    GF61 trig = TOLOAD(&trigH61[height_trigs + line_u * G_H * 2 + me]);
#endif

    if (line_u == 0) {
      reverse2(megaLds, u);
      pairSq2_special(u, trig);
      reverse2(megaLds, u);
    } else {
      revCrossLine(megaLds, u);
      pairSq(NH / 2, u, u + NH / 2, trig, false);
      revCrossLine(megaLds, u);
    }

    fft_HEIGHT2(megaLds, u, trigH61, 2, lowMe);
    writeTailFusedLine(u, scrB61, transPos(line, MIDDLE, WIDTH), lowMe);

    megaPublish(rdy, MEGA_TAILFLAG + g, epoch);

  } else {
    // ------------------------- MidOut: fftmiddleout.cl fftMiddleOutGF61 body -------------------------
    u32 gxo  = desc & 31;
    u32 gyo  = (desc >> 5) & 63;
    u32 slot = (desc >> 11) & 2047;

    // Threads 0..63 each spin one host-verified tail flag from the map's flag table.
    if (me < 64) {
      megaSpinWait(rdy, map[MEGA_FLAGTAB + slot * 64 + me], epoch, 2, dbg, MEGA_FAIL_PTR);
    }
    bar();
    read_mem_fence(CLK_GLOBAL_MEM_FENCE);
#if MEGA_DEBUG
    if (megaFail) { return; }
#endif

    GF61 u[MIDDLE];
    u32 SIZEY = OUT_WG / OUT_SIZEX;
    u32 mx = me % OUT_SIZEX;
    u32 my = me / OUT_SIZEX;
    u32 x = gxo * OUT_SIZEX + mx;
    u32 y = gyo * SIZEY + my;

    readMiddleOutLine(u, scrB61, y, x);
    middleMul(u, x, trigM61);
    fft_MIDDLE(u);
    middleMul2(u, y, x, trigM61);

    local Z61 *lds = (local Z61 *) megaLds;
    middleShuffle(lds, u, OUT_WG, OUT_SIZEX);
    writeMiddleOutLine(out61 + me, u, gyo, gxo);
    // No publish in v1: the following carryFused launch is a natural barrier.
  }
}

#endif // NTT_GF61 && MEGAPASS61

#endif // MEGAPASS31 || MEGAPASS61
