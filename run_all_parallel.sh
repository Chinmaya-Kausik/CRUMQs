#!/bin/bash

source ./unans_env/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true
export LITELLM_API_KEY="${LITELLM_API_KEY}"
export LITELLM_IP="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

NEUCLIR_ARGS="--database_dir=./_database/neuclir_test --report_requests_path=./_database/test_requests_neuclir.jsonl --num_articles_to_crawl=25 --num_topics=25 --dataset_test_size=100 --save_dir=./generated_data/mh_neuclir --max_q_prompts_per_request=100 --max_unanswerable_qs_per_request=100 --max_node_dicts=100 --max_per_split=25"

# Remaining NeuCLIR IDs (skip completed: 300,303,308,309,310,334,335)
# Also skip 343 if it's currently in progress
REMAINING_IDS=(343 351 352 365 367 372 373 377 380 382 383 388)

# Number of parallel workers
NUM_WORKERS=3

run_request() {
    local id=$1
    echo "$(date): Starting NeuCLIR id=$id"
    python ./src/crumqs_generation/run.py $NEUCLIR_ARGS --exp_name=$id --ids_to_run $id \
        > ./generated_data/mh_neuclir/${id}/run.log 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "$(date): COMPLETED NeuCLIR id=$id"
    else
        echo "$(date): FAILED NeuCLIR id=$id (exit $status)"
    fi
}

echo "========== PARALLEL NEUCLIR START: $(date) =========="
echo "Running ${#REMAINING_IDS[@]} requests with $NUM_WORKERS workers"

# Run in parallel with GNU parallel-style batching
idx=0
while [ $idx -lt ${#REMAINING_IDS[@]} ]; do
    pids=()
    for ((w=0; w<NUM_WORKERS && idx<${#REMAINING_IDS[@]}; w++, idx++)); do
        id=${REMAINING_IDS[$idx]}
        run_request $id &
        pids+=($!)
    done
    # Wait for this batch to finish
    for pid in "${pids[@]}"; do
        wait $pid
    done
done

echo "========== PARALLEL NEUCLIR DONE: $(date) =========="

echo "========== TRECRAG START: $(date) =========="
bash ./src/crumqs_generation/run_trecrag.sh
echo "========== TRECRAG DONE: $(date) =========="
