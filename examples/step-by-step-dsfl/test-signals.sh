#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEFAULT_PYTHON_BIN=$SCRIPT_DIR/.venv/bin/python
if [ ! -x "$DEFAULT_PYTHON_BIN" ]; then
    DEFAULT_PYTHON_BIN=$SCRIPT_DIR/../../.venv/bin/python
fi
PYTHON_BIN=${PYTHON_BIN:-"$DEFAULT_PYTHON_BIN"}
STARTUP_WAIT=${STARTUP_WAIT:-20}
EXIT_TIMEOUT=${EXIT_TIMEOUT:-20}
READY_PATTERN=${READY_PATTERN:-device:}

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    echo "Run 'uv sync' in $SCRIPT_DIR first, or set PYTHON_BIN." >&2
    exit 1
fi

print_log_and_fail() {
    message=$1
    log_file=$2
    echo "$message" >&2
    echo "--- begin log ---" >&2
    cat "$log_file" >&2
    echo "--- end log ---" >&2
    exit 1
}

cleanup_pid() {
    pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

wait_for_ready() {
    pid=$1
    log_file=$2
    seconds=0
    while ! grep -q "$READY_PATTERN" "$log_file" 2>/dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then
            print_log_and_fail "process $pid exited before reaching ready state" "$log_file"
        fi
        if [ "$seconds" -ge "$STARTUP_WAIT" ]; then
            print_log_and_fail "process $pid did not reach ready state within ${STARTUP_WAIT}s" "$log_file"
        fi
        sleep 1
        seconds=$((seconds + 1))
    done
}

wait_for_exit() {
    pid=$1
    signal_name=$2
    log_file=$3
    seconds=0
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$seconds" -ge "$EXIT_TIMEOUT" ]; then
            cleanup_pid "$pid"
            print_log_and_fail "$signal_name test failed: process $pid did not exit within ${EXIT_TIMEOUT}s" "$log_file"
        fi
        sleep 1
        seconds=$((seconds + 1))
    done
}

assert_clean_shutdown() {
    signal_name=$1
    log_file=$2

    if grep -Eq 'Traceback \(most recent call last\):|BrokenPipeError:' "$log_file"; then
        print_log_and_fail "$signal_name test failed: traceback detected during shutdown" "$log_file"
    fi
}

run_signal_test() {
    signal_name=$1
    log_file=$(mktemp "${TMPDIR:-/tmp}/step-by-step-dsfl-signal.XXXXXX")

    "$PYTHON_BIN" "$SCRIPT_DIR/main.py" \
        --global-round 1000 \
        --epochs 100000 \
        --num-clients 10 \
        --sample-ratio 1.0 \
        --num-parallels 2 >"$log_file" 2>&1 &
    pid=$!

    printf "Started step-by-step-dsfl with PID %s for %s test\n" "$pid" "$signal_name"
    wait_for_ready "$pid" "$log_file"
    kill "-$signal_name" "$pid"
    wait_for_exit "$pid" "$signal_name" "$log_file"
    wait "$pid" 2>/dev/null || true
    assert_clean_shutdown "$signal_name" "$log_file"
    rm -f "$log_file"
    printf "%s test passed\n" "$signal_name"
}

run_signal_test INT
run_signal_test TERM
