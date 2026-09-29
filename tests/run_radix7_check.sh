#!/usr/bin/env bash
# GPU check of the 7 * 2^k FFT3161 transforms (radix-7 middle): every 7 * 2^k shape against GMP
# (single, fused and Lucas-Lehmer squarings and a prepared multiplication), PRP throughput of the
# 3.5M shapes against the 4M power-of-two transform at M136279841, Gerbicz-Li checks at every block
# on 3.5M, and full PRPs with proof of the Mersenne primes M3021377 (long carry) and M13466917
# (fused carry) on the 896K shape 256:7:256.
#
# usage: tests/run_radix7_check.sh [device] [output-dir]
#   PRMERS_CHECK_SECONDS=<s>   seconds per prmers run (default 90)
#   PRMERS_CHECK_QUICK=1       skip the full PRPs
#
# Build first with ./build_with_aevum_engine.sh. Writes summary.txt and one log per step.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
OUT="${2:-$ROOT/tests/radix7-check}"
P_SPEED=136279841
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

# 1. Engine level: each 7 * 2^k shape against GMP at exponents across its range of bits per word.
while read -r spec exponents; do
    log="$OUT/gmp_${spec//:/_}.log"
    if AEVUM_TEST_FFT="$spec" make -C "$ROOT/third_party/aevum" test-square-loop-gpu AEVUM_TEST_DEVICE="$DEVICE" \
            AEVUM_TEST_EXPONENTS="$exponents" > "$log" 2>&1; then
        note "PASS  $spec matches GMP at $exponents"
    else
        note "FAIL  $spec against GMP ($(basename "$log"))"
        failures=$((failures + 1))
    fi
    grep -h -m1 -E "Fused square loop|self-test" "$log" | tee -a "$SUMMARY"
    grep -h -E "ms/iter" "$log" | tee -a "$SUMMARY"
done <<'EOF'
1:256:7:256:202 3021377:200 13466917:100 30402457:100
1:512:7:256:202 57885161:60
1:256:7:512:202 57885161:60
1:512:7:512:202 136279841:40
1:1K:7:256:202 136279841:40
1:1K:7:512:202 250000013:20
1:512:7:1K:202 250000013:20
EOF

# 2. PRP throughput at M136279841: the automatic 4M transform and the two 3.5M shapes.
ips_4m="$(speed_run auto_4M -- -aevum)"
ips_512="$(speed_run r7_512_7_512 -- -aevum -aevum-fft 1:512:7:512:202)"
ips_1k="$(speed_run r7_1K_7_256 -- -aevum -aevum-fft 1:1K:7:256:202)"
grep -h -m1 "Aevum auto FFT" "$OUT/speed_auto_4M.log" | tee -a "$SUMMARY"
note "M$P_SPEED PRP iterations/s: 4M (automatic) $ips_4m, 3.5M 512:7:512 $ips_512, 3.5M 1K:7:256 $ips_1k"
note "3.5M speed-up: $(awk -v a="$ips_512" -v c="$ips_1k" -v b="$ips_4m" 'BEGIN {if (b > 0) printf "512:7:512 x%.3f, 1K:7:256 x%.3f", a / b, c / b; else print "n/a"}')"

# 3. Gerbicz-Li checks at every block on the 3.5M transform.
clean_state "$P_SPEED"
(cd "$WORK" && run_for "$SECONDS_PER_RUN" "$PRMERS" "$P_SPEED" -aevum -aevum-fft 1:512:7:512:202 -checklevel 1 -proof 0 -d "$DEVICE" --noask) \
    > "$OUT/gerbicz_3.5M.log" 2>&1
clean_state "$P_SPEED"
pass_if "Gerbicz-Li checks pass on 3.5M" "$OUT/gerbicz_3.5M.log" "Check passed"
fail_if "no Gerbicz-Li error on 3.5M" "$OUT/gerbicz_3.5M.log" "Check FAILED"
note "Gerbicz-Li checks passed: $(grep -c "Check passed" "$OUT/gerbicz_3.5M.log")"

# 4. Full PRPs with proof of Mersenne primes on the 896K shape.
if [[ "${PRMERS_CHECK_QUICK:-0}" != "1" ]]; then
    for p in 3021377 13466917; do
        clean_state "$p"
        (cd "$WORK" && "$PRMERS" "$p" -aevum -aevum-fft 1:256:7:256:202 -d "$DEVICE" --noask) > "$OUT/prp_${p}_896K.log" 2>&1
        pass_if "M$p is a probable prime on 256:7:256" "$OUT/prp_${p}_896K.log" "2^$p - 1 is a probable prime"
        pass_if "M$p proof generated on the GPU" "$OUT/prp_${p}_896K.log" "Proof generated on the GPU"
        fail_if "M$p without Gerbicz-Li errors" "$OUT/prp_${p}_896K.log" "Check FAILED"
        grep -h -E "is a probable prime|Proof generated" "$OUT/prp_${p}_896K.log" | tee -a "$SUMMARY"
        clean_state "$p"
    done
fi

note "$([[ $failures -eq 0 ]] && echo "All checks passed" || echo "$failures check(s) failed"). Summary: $SUMMARY"
exit $((failures == 0 ? 0 : 1))
