# Helpers shared by the GPU check scripts (sourced, not run).
# The caller sets OUT, WORK, SUMMARY, PRMERS, DEVICE, P_SPEED, SECONDS_PER_RUN and failures=0.

note() { echo "$*" | tee -a "$SUMMARY"; }
pass_if() { # label, file, pattern
    if grep -qF -- "$3" "$2"; then note "PASS  $1"; else note "FAIL  $1 (see $(basename "$2"))"; failures=$((failures + 1)); fi
}
fail_if() { # label, file, pattern: the check fails when the pattern is present
    if grep -qF -- "$3" "$2"; then note "FAIL  $1 (see $(basename "$2"))"; failures=$((failures + 1)); else note "PASS  $1"; fi
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
speed_run() { # label, extra environment assignments..., -- , prmers arguments...: prints the median IPS
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
