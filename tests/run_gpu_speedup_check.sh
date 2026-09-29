#!/usr/bin/env bash
# GPU check of the fused Aevum square loop, GPU proof generation and Marin 5*2^k transforms,
# plus a PRP speed comparison of the fused loop against the per-squaring path.
#
# usage: tests/run_gpu_speedup_check.sh [device] [output-dir]
#   PRMERS_CHECK_EXPONENT=<p>   exponent for the speed runs (default 136279841)
#   PRMERS_CHECK_SECONDS=<s>    seconds per speed run (default 90)
#   PRMERS_CHECK_QUICK=1        skip the full PRP and LL runs of M3021377
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

note() { echo "$*" | tee -a "$SUMMARY"; }
pass_if() { # label, file, pattern
    if grep -qF -- "$3" "$2"; then note "PASS  $1"; else note "FAIL  $1 (see $(basename "$2"))"; failures=$((failures + 1)); fi
}
clean_state() { # exponent: remove checkpoints and proof data left by earlier runs
    rm -rf "$WORK"/m_"$1".ckpt* "$WORK"/llunsafe_m_"$1".ckpt* "$WORK/$1"
}
median_ips() { # log: median of the per-interval IPS values, skipping the first two (warm-up)
    tr '\r' '\n' < "$1" | grep -o 'IPS: [0-9.]*' | awk '{print $2}' | tail -n +3 | sort -n |
        awk '{v[NR] = $1} END {if (NR == 0) print 0; else if (NR % 2) print v[(NR + 1) / 2]; else print (v[NR / 2] + v[NR / 2 + 1]) / 2}'
}
run_for() { # seconds, command...: stop the command after that much wall time (exit status 124)
    local seconds="$1"; shift
    case "$(uname -s)" in
    *_NT*)
        # MSYS2 and Cygwin signals do not reach a native Windows program such as prmers.exe,
        # so end it with taskkill on its Windows process id.
        "$@" &
        local pid=$! end=$((SECONDS + seconds))
        while kill -0 "$pid" 2>/dev/null && [[ $SECONDS -lt $end ]]; do sleep 1; done
        if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; return; fi
        MSYS2_ARG_CONV_EXCL='*' taskkill /F /T /PID "$(cat "/proc/$pid/winpid")" > /dev/null
        wait "$pid"
        return 124
        ;;
    *) timeout -s INT "$seconds" "$@" ;;
    esac
}
speed_run() { # label, extra environment assignments..., -- , prmers arguments...
    local label="$1"; shift
    local env_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do env_args+=("$1"); shift; done
    shift
    clean_state "$P_SPEED"
    (cd "$WORK" && for a in ${env_args[@]+"${env_args[@]}"}; do export "$a"; done &&
        run_for "$SECONDS_PER_RUN" "$PRMERS" "$P_SPEED" "$@" -proof 0 -d "$DEVICE" --noask) \
        > "$OUT/speed_$label.log" 2>&1
    clean_state "$P_SPEED"
    median_ips "$OUT/speed_$label.log"
}

note "PrMers $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown), device $DEVICE, $(date)"

# 1. Engine level: square_loop against GMP, and single square_mul calls versus the fused loop.
if make -C "$ROOT/third_party/aevum" test-square-loop-gpu AEVUM_TEST_DEVICE="$DEVICE" \
        AEVUM_TEST_EXPONENTS="1362763:100 2976221:200 30402457:100 $P_SPEED:30" > "$OUT/square_loop.log" 2>&1; then
    note "PASS  square_loop matches GMP (square_loop.log)"
else
    note "FAIL  square_loop test (square_loop.log)"
    failures=$((failures + 1))
fi
grep -E "Fused square loop|self-test|ms/iter" "$OUT/square_loop.log" | tee -a "$SUMMARY"

# 2. PRP throughput at the wavefront: the fused loop, the previous per-squaring path (same
#    binary, AEVUM_DISABLE_FUSED_LOOP=1) and Marin for reference.
ips_fused="$(speed_run aevum_fused -- -aevum)"
ips_plain="$(speed_run aevum_unfused AEVUM_DISABLE_FUSED_LOOP=1 -- -aevum)"
ips_marin="$(speed_run marin -- -engine-marin)"
grep -h -m1 "Running on device" "$OUT/speed_marin.log" | tee -a "$SUMMARY"
grep -h -m1 -E "Fused square loop" "$OUT/speed_aevum_fused.log" | tee -a "$SUMMARY"
note "M$P_SPEED PRP iterations/s: Aevum fused $ips_fused, Aevum unfused $ips_plain, Marin $ips_marin"
note "Fused loop speed-up: $(awk -v a="$ips_fused" -v b="$ips_plain" 'BEGIN {if (b > 0) printf "x%.3f", a / b; else print "n/a"}')"

# 3. End to end on the Mersenne prime M3021377: PRP with a GPU-generated proof, then LL.
if [[ "${PRMERS_CHECK_QUICK:-0}" != "1" ]]; then
    clean_state 3021377
    (cd "$WORK" && "$PRMERS" 3021377 -aevum -d "$DEVICE" --noask) > "$OUT/prp_3021377.log" 2>&1
    pass_if "Aevum PRP: M3021377 is a probable prime" "$OUT/prp_3021377.log" "2^3021377 - 1 is a probable prime"
    pass_if "Aevum PRP: proof generated on the GPU" "$OUT/prp_3021377.log" "Proof generated on the GPU"
    grep -h -E "Proof generated|Check FAILED" "$OUT/prp_3021377.log" | tee -a "$SUMMARY"
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
