#!/bin/bash
#
trials=$1
alpha=$2
offset=0
proc=4
function run() {
    batch=$1

    for seed in $(seq ${batch} ${proc} ${trials}); do
        python -m wcs_exp.drug_discovery.DPP ${seed} ${alpha}
    done
    touch job_done_statuses/finished_${batch}.txt
}

if [ -d "job_done_statuses" ]; then
    rm -r job_done_statuses/
fi
mkdir -p job_done_statuses


for batch in {0..3}; do
    start_idx=$((${batch} + ${offset}))
    echo "running $start_idx"
    run ${start_idx} &
done

