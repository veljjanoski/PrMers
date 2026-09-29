#!/usr/bin/env bash
# Tunes the Aevum engine for this GPU. It times the FFT3161 shapes for an exponent and Aevum's
# kernel settings (third_party/aevum/tests/engine_tune_gpu.cpp), measures prmers PRP throughput
# with the default and the tuned configuration, checks the tuned configuration with Gerbicz-Li
# checks at that exponent and with a full PRP of M3021377, and writes the options to
# aevum-tuned.cfg:   prmers <exponent> -config <output-dir>/aevum-tuned.cfg
#
# usage: tests/run_aevum_tune.sh [device] [output-dir]
#   PRMERS_TUNE_EXPONENT=<p>   exponent to tune for (default 136279841)
#   PRMERS_TUNE_MODE=quick     skip the cache-hint settings and those used only when INPLACE=0
#   PRMERS_CHECK_SECONDS=<s>   seconds per prmers run (default 90)
#   PRMERS_CHECK_QUICK=1       skip the full PRP of M3021377
#
# Build first with ./build_with_aevum_engine.sh. Writes summary.txt and one log per step.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
OUT="${2:-$ROOT/tests/aevum-tune}"
P_SPEED="${PRMERS_TUNE_EXPONENT:-136279841}"
MODE="${PRMERS_TUNE_MODE:-full}"
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
rm -f "$OUT/aevum-tuned.cfg"
failures=0

source "$ROOT/tests/gpu_check_common.sh"

note "PrMers $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown), device $DEVICE, M$P_SPEED, $MODE tuning, $(date)"

# 1. The sweep: every configuration is checked against GMP before it is timed.
make -C "$ROOT/third_party/aevum" tune-gpu AEVUM_TEST_DEVICE="$DEVICE" AEVUM_TUNE_EXPONENT="$P_SPEED" \
    AEVUM_TUNE_MODE="$MODE" > "$OUT/tune.log" 2>&1
options="$(sed -n 's/^tune: prmers options: //p' "$OUT/tune.log")"
if [[ -z "$options" ]]; then
    note "FAIL  tuning sweep (tune.log)"
    tail -n 20 "$OUT/tune.log"
    exit 1
fi
grep -E "^tune: (default|tuned) " "$OUT/tune.log" | tee -a "$SUMMARY"
wrong="$(grep -c "WRONG RESULT" "$OUT/tune.log")"
note "Configurations with a wrong result (excluded): $wrong"
use="$(echo "$options" | sed -n 's/.*-aevum-use \([^ ]*\).*/\1/p')"

# 2. PRP throughput at the exponent, default then tuned configuration.
ips_default="$(speed_run default -- -aevum)"
# shellcheck disable=SC2086  # options holds several arguments
ips_tuned="$(speed_run tuned -- -aevum $options)"
note "M$P_SPEED PRP iterations/s: default $ips_default, tuned $ips_tuned"
note "Tuned speed-up: $(awk -v a="$ips_tuned" -v b="$ips_default" 'BEGIN {if (b > 0) printf "x%.3f", a / b; else print "n/a"}')"

# 3. Gerbicz-Li checks at every block (-checklevel 1) with the tuned configuration.
clean_state "$P_SPEED"
# shellcheck disable=SC2086
(cd "$WORK" && run_for "$SECONDS_PER_RUN" "$PRMERS" "$P_SPEED" -aevum $options -checklevel 1 -proof 0 -d "$DEVICE" --noask) \
    > "$OUT/gerbicz_tuned.log" 2>&1
clean_state "$P_SPEED"
pass_if "Gerbicz-Li checks pass with the tuned configuration" "$OUT/gerbicz_tuned.log" "Check passed"
fail_if "no Gerbicz-Li error with the tuned configuration" "$OUT/gerbicz_tuned.log" "Check FAILED"
note "Gerbicz-Li checks passed: $(grep -c "Check passed" "$OUT/gerbicz_tuned.log")"

# 4. A full PRP with proof of the Mersenne prime M3021377 with the tuned kernel settings (its
#    own, smaller FFT: the tuned shape is for the size of M$P_SPEED).
if [[ "${PRMERS_CHECK_QUICK:-0}" != "1" ]]; then
    clean_state 3021377
    (cd "$WORK" && "$PRMERS" 3021377 -aevum ${use:+-aevum-use "$use"} -d "$DEVICE" --noask) > "$OUT/prp_3021377_tuned.log" 2>&1
    pass_if "M3021377 is a probable prime with the tuned settings" "$OUT/prp_3021377_tuned.log" "2^3021377 - 1 is a probable prime"
    pass_if "proof generated on the GPU" "$OUT/prp_3021377_tuned.log" "Proof generated on the GPU"
    clean_state 3021377
fi

if [[ $failures -eq 0 ]]; then
    echo "$options" > "$OUT/aevum-tuned.cfg"
    note "Tuned options: $options"
    note "Written to $OUT/aevum-tuned.cfg; use it with: prmers <exponent> -config $OUT/aevum-tuned.cfg"
    note "(the FFT shape is for exponents with the transform size of M$P_SPEED)"
fi
note "$([[ $failures -eq 0 ]] && echo "All checks passed" || echo "$failures check(s) failed, no configuration written"). Summary: $SUMMARY"
exit $((failures == 0 ? 0 : 1))
