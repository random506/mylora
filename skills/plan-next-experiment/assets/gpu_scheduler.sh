#!/bin/bash
# Generated from the plan-next-experiment skill.
# Plan: __PLAN_PATH__

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT" || {
    echo "[scheduler] Could not enter repository root: $REPO_ROOT" >&2
    exit 1
}

MIN_VRAM_GB=47
POLL_INTERVAL=60
MAX_WAIT_HOURS=72
MAX_CONCURRENT_GPUS=2

# Format: "<required GPU count> | <command>"
TRAIN_TASK_LIST=(
    # __TRAIN_TASKS__
)

EVAL_TASK_LIST=(
    # __EVAL_TASKS__
)

fail() {
    echo "[scheduler] $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Missing command: $1"
}

require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || fail "Missing environment variable: $name"
}

require_file() {
    [[ -f "$1" ]] || fail "Missing file: $1"
}

require_dir() {
    [[ -d "$1" ]] || fail "Missing directory: $1"
}

for command_name in python nvidia-smi bc awk sed date cut tr; do
    require_command "$command_name"
done

require_file ".env"
set -a
# shellcheck disable=SC1091
source ".env"
set +a

for env_name in HF_CACHE_DIR HF_DATASETS_DIR STATS_DIR; do
    require_env "$env_name"
done

[[ "$HF_CACHE_DIR" == */ ]] \
    || fail "HF_CACHE_DIR must end with '/' because project paths use string concatenation."

# __EXPERIMENT_PREFLIGHT__

MAX_WAIT_SEC=$(echo "$MAX_WAIT_HOURS * 3600" | bc | cut -d. -f1)
START_TIME=$(date +%s)

get_free_gpus() {
    local min_mb
    min_mb=$(echo "$MIN_VRAM_GB * 1024" | bc | cut -d. -f1)
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v min="$min_mb" '
            {
                id=$1; free=$2
                gsub(/ /,"",id); gsub(/ /,"",free)
                if (free+0 >= min+0) print id
            }
        '
}

gpu_status_str() {
    nvidia-smi --query-gpu=index,memory.free,memory.total \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '
            {
                id=$1; free=$2; total=$3
                gsub(/ /,"",id); gsub(/ /,"",free); gsub(/ /,"",total)
                printf " GPU%s:%.0f/%.0fGB", id, free/1024, total/1024
            }
        '
}

run_stage() {
    local stage_name="$1"
    shift
    local entries=("$@")
    local total=${#entries[@]}

    if (( total == 0 )); then
        echo "[scheduler] Stage '$stage_name' has no tasks; skipping."
        return 0
    fi

    local -a task_gpus=()
    local -a task_commands=()
    local -a task_states=()
    local -a task_pids=()
    local -a task_gpu_ids=()
    local entry gpu_count task_command
    local index

    for entry in "${entries[@]}"; do
        gpu_count="${entry%%|*}"
        task_command="${entry#*|}"
        gpu_count=$(echo "$gpu_count" | tr -d ' ')
        # shellcheck disable=SC2001
        task_command=$(echo "$task_command" | sed 's/^[[:space:]]*//')
        [[ "$gpu_count" =~ ^[1-9][0-9]*$ ]] \
            || fail "Invalid GPU count in stage '$stage_name': $entry"
        (( gpu_count <= MAX_CONCURRENT_GPUS )) \
            || fail "Task needs $gpu_count GPUs but the limit is $MAX_CONCURRENT_GPUS: $task_command"
        task_gpus+=("$gpu_count")
        task_commands+=("$task_command")
        task_states+=("pending")
        task_pids+=("")
        task_gpu_ids+=("")
    done

    echo "[scheduler] Starting stage '$stage_name' with $total tasks."
    for (( index=0; index<total; index++ )); do
        echo "  [$stage_name:$((index+1))] ${task_gpus[index]} GPU(s) | ${task_commands[index]}"
    done

    local failed=0
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        local now elapsed
        now=$(date +%s)
        elapsed=$((now - START_TIME))
        local active_gpus=0
        local pending=0
        local running=0
        local -a active_ids=()
        local pid assigned

        for (( index=0; index<total; index++ )); do
            case "${task_states[index]}" in
                pending)
                    pending=$((pending + 1))
                    ;;
                running)
                    pid="${task_pids[index]}"
                    if kill -0 "$pid" 2>/dev/null; then
                        running=$((running + 1))
                        active_gpus=$((active_gpus + task_gpus[index]))
                        IFS=',' read -ra assigned <<< "${task_gpu_ids[index]}"
                        active_ids+=("${assigned[@]}")
                    else
                        if wait "$pid"; then
                            task_states[index]="succeeded"
                            echo "[scheduler] [$stage_name:$((index+1))] succeeded."
                        else
                            task_states[index]="failed"
                            failed=1
                            echo "[scheduler] [$stage_name:$((index+1))] failed: ${task_commands[index]}" >&2
                        fi
                    fi
                    ;;
            esac
        done

        if (( pending == 0 && running == 0 )); then
            break
        fi

        if (( pending > 0 && elapsed > MAX_WAIT_SEC )); then
            fail "GPU wait timed out after ${MAX_WAIT_HOURS}h."
        fi

        local -a free_ids=()
        mapfile -t free_ids < <(get_free_gpus)
        local -a used_ids=()
        local -a available=()
        local need remaining gpu_id used skip
        local cuda_ids

        echo "[scheduler] [$stage_name] check $attempt:$(gpu_status_str) pending=$pending running_gpus=$active_gpus/$MAX_CONCURRENT_GPUS"

        for (( index=0; index<total; index++ )); do
            [[ "${task_states[index]}" == "pending" ]] || continue
            need=${task_gpus[index]}
            remaining=$((MAX_CONCURRENT_GPUS - active_gpus - ${#used_ids[@]}))
            (( need <= remaining )) || continue

            available=()
            for gpu_id in "${free_ids[@]}"; do
                skip=0
                for used in "${active_ids[@]}" "${used_ids[@]}"; do
                    if [[ "$gpu_id" == "$used" ]]; then
                        skip=1
                        break
                    fi
                done
                (( skip == 1 )) || available+=("$gpu_id")
            done
            (( ${#available[@]} >= need )) || continue

            assigned=("${available[@]:0:need}")
            used_ids+=("${assigned[@]}")
            cuda_ids=$(IFS=','; echo "${assigned[*]}")
            echo "[scheduler] [$stage_name:$((index+1))] CUDA_VISIBLE_DEVICES=$cuda_ids ${task_commands[index]}"
            (
                export CUDA_VISIBLE_DEVICES="$cuda_ids"
                eval "${task_commands[index]}"
            ) &
            task_pids[index]=$!
            task_gpu_ids[index]="$cuda_ids"
            task_states[index]="running"
        done

        sleep "$POLL_INTERVAL"
    done

    (( failed == 0 ))
}

if ! run_stage "training" "${TRAIN_TASK_LIST[@]}"; then
    fail "Training failed; evaluation will not start."
fi

if ! run_stage "evaluation" "${EVAL_TASK_LIST[@]}"; then
    fail "Evaluation failed."
fi

echo "[scheduler] All selected stages completed successfully."
