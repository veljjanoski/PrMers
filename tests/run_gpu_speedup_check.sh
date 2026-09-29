#!/usr/bin/env bash
# GPU check of Aevum's chained squarings (register lead cache) against GMP, GPU proof generation
# and Marin 5*2^k transforms, plus PRP speed at the wavefront: the default Aevum engine, without
# the lead cache (AEVUM_REG_LEAD_CACHE=0), on a single queue (AEVUM_TYPE4_MULTI_Q=0) and Marin.
#
# usage: tests/run_gpu_speedup_check.sh [device] [output-dir]
#   PRMERS_CHECK_EXPONENT=<p>   exponent for the speed runs (default 136279841)
#   PRMERS_CHECK_SECONDS=<s>    seconds per speed run (default 90)
#   PRMERS_CHECK_QUICK=1        skip the full PRP of M6972593 and LL of M3021377
#
# Build first with ./build_with_aevum_engine.sh. Writes summary.txt and one log per step.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
OUT="${2:-$ROOT/tests/gpu-speedup-check}"
P_SPEED="${PRMERS_CHECK_EXPONENT:-136279841}"
SECONDS_PER_RUN="${PRMERS_CHECK_SECONDS:-90}"
PRMERS="$ROOT/prmers"

if [[ ! -x "$PRMERS" ]]; then
    echo "prmers is not built: run ./build_with_aevum_engine.sh first" >&2
    exit 2
fi

mkdir -p "$OUT/work"
OUT="$(cd "$OUT" && pwd)"
WORK="$OUT/work"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"
failures=0

source "$ROOT/tests/gpu_check_common.sh"

note "PrMers $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown), device $DEVICE, $(date)"

# 1. Engine level: chained square_mul calls, a prepared multiplication after a chain and
#    Lucas-Lehmer steps against GMP, with the time per squaring.
if make -C "$ROOT/third_party/aevum" test-square-loop-gpu AEVUM_TEST_DEVICE="$DEVICE" \
        AEVUM_TEST_EXPONENTS="1362763:100 2976221:200 30402457:100 $P_SPEED:30" > "$OUT/square_chain.log" 2>&1; then
    note "PASS  chained squarings match GMP (square_chain.log)"
else
    note "FAIL  chained squaring test (square_chain.log)"
    failures=$((failures + 1))
fi
grep -E "ms/iter|MISMATCH" "$OUT/square_chain.log" | tee -a "$SUMMARY"

# 2. PRP throughput at the wavefront: the default engine, without the register lead cache, on a
#    single queue, and Marin for reference.
ips_default="$(speed_run aevum_default -- -aevum)"
ips_nolead="$(speed_run aevum_no_lead_cache AEVUM_REG_LEAD_CACHE=0 -- -aevum)"
ips_singleq="$(speed_run aevum_single_queue AEVUM_TYPE4_MULTI_Q=0 -- -aevum)"
ips_marin="$(speed_run marin -- -engine-marin)"
grep -h -m1 "Running on device" "$OUT/speed_marin.log" | tee -a "$SUMMARY"
grep -h -m1 "adapter active" "$OUT/speed_aevum_default.log" | tee -a "$SUMMARY"
note "M$P_SPEED PRP iterations/s: Aevum default $ips_default, without lead cache $ips_nolead, single queue $ips_singleq, Marin $ips_marin"
note "Lead cache speed-up: $(awk -v a="$ips_default" -v b="$ips_nolead" 'BEGIN {if (b > 0) printf "x%.3f", a / b; else print "n/a"}'), multi-queue speed-up: $(awk -v a="$ips_default" -v b="$ips_singleq" 'BEGIN {if (b > 0) printf "x%.3f", a / b; else print "n/a"}')"
for label in aevum_default aevum_no_lead_cache aevum_single_queue; do
    fail_if "no Gerbicz-Li error ($label)" "$OUT/speed_$label.log" "Check FAILED"
done

# 3. End to end on Mersenne primes: an Aevum PRP with a GPU-generated proof of M6972593 on 1M words
#    (Aevum refuses ordinary PRP on 512K words or fewer), then an Aevum LL of M3021377.
if [[ "${PRMERS_CHECK_QUICK:-0}" != "1" ]]; then
    clean_state 6972593
    (cd "$WORK" && "$PRMERS" 6972593 -aevum -aevum-fft 1:512:2:512 -d "$DEVICE" --noask) > "$OUT/prp_6972593.log" 2>&1
    pass_if "Aevum PRP: M6972593 is a probable prime" "$OUT/prp_6972593.log" "2^6972593 - 1 is a probable prime"
    pass_if "Aevum PRP: proof generated on the GPU" "$OUT/prp_6972593.log" "Proof generated on the GPU"
    fail_if "Aevum PRP: no Gerbicz-Li error" "$OUT/prp_6972593.log" "Check FAILED"
    grep -h -E "Proof generated|Check FAILED" "$OUT/prp_6972593.log" | tee -a "$SUMMARY"
    clean_state 6972593
    clean_state 3021377
    (cd "$WORK" && "$PRMERS" 3021377 -llunsafe -aevum -d "$DEVICE" --noask) > "$OUT/ll_3021377.log" 2>&1
    pass_if "Aevum LL: M3021377 is prime" "$OUT/ll_3021377.log" "2^3021377 - 1 is prime"
    clean_state 3021377
fi

# 4. Marin 5*2^k lengths (radix-5 barrier fix): M110503 runs on 5120 words.
clean_state 110503
(cd "$WORK" && "$PRMERS" 110503 -engine-marin -proof 0 -d "$DEVICE" --noask) > "$OUT/prp_110503_marin.log" 2>&1
pass_if "Marin 5120-word PRP: M110503 is a probable prime" "$OUT/prp_110503_marin.log" "2^110503 - 1 is a probable prime"
clean_state 110503

note "$([[ $failures -eq 0 ]] && echo "All checks passed" || echo "$failures check(s) failed"). Summary: $SUMMARY"
exit $((failures == 0 ? 0 : 1))
