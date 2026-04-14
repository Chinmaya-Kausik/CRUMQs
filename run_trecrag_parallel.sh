#!/bin/bash

source ./unans_env/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true
export LITELLM_API_KEY="${LITELLM_API_KEY}"
export LITELLM_IP="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

TRECRAG_ARGS="--database_dir=./_database/trecrag2025 --report_requests_path=./_database/test_requests_trecrag2025.jsonl --num_articles_to_crawl=25 --num_topics=25 --dataset_test_size=100 --save_dir=./generated_data/mh_trecrag --max_q_prompts_per_request=100 --max_unanswerable_qs_per_request=100 --max_node_dicts=100 --max_per_split=25"

ALL_IDS=(6 7 9 14 18 23 24 25 28 37 41 43 46 47 48 49 53 55 56 58 60 66 70 72 80 84 85 93 97 100 102 125 129 140 144 147 149 161 176 182 183 187 188 191 202 203 213 219 221 224 225 231 233 245 247 258 273 278 300 323 333 341 346 349 360 372 388 394 407 426 467 469 477 497 499 502 503 515 526 528 568 571 582 588 592 598 606 707 838 839 847 851 882 889 897 915 971 983 986 988 1001)

# Skip already-completed requests
REMAINING_IDS=()
for id in "${ALL_IDS[@]}"; do
    dir="./generated_data/mh_trecrag/$id/_data_multihop"
    if [ -d "$dir" ]; then
        count=$(ls "$dir"/*.pkl 2>/dev/null | wc -l)
        if [ "$count" -ge 5 ]; then
            continue
        fi
    fi
    REMAINING_IDS+=($id)
done

NUM_WORKERS=2

run_request() {
    local id=$1
    echo "$(date): Starting TRECRAG id=$id"
    mkdir -p ./generated_data/mh_trecrag/$id
    python ./src/crumqs_generation/run.py $TRECRAG_ARGS --exp_name=$id --ids_to_run $id \
        > ./generated_data/mh_trecrag/${id}/run.log 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "$(date): COMPLETED TRECRAG id=$id"
    else
        echo "$(date): FAILED TRECRAG id=$id (exit $status)"
    fi
}

echo "========== PARALLEL TRECRAG START: $(date) =========="
echo "Running ${#REMAINING_IDS[@]} remaining requests with $NUM_WORKERS workers"

idx=0
while [ $idx -lt ${#REMAINING_IDS[@]} ]; do
    pids=()
    for ((w=0; w<NUM_WORKERS && idx<${#REMAINING_IDS[@]}; w++, idx++)); do
        id=${REMAINING_IDS[$idx]}
        run_request $id &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid
    done
done

echo "========== PARALLEL TRECRAG DONE: $(date) =========="
