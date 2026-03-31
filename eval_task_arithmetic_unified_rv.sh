MERGE_CONFIGS=("imagenet100_mt_bv_rv" "imagenet100_mt_bv_rv_fare4" "imagenet100_mt_bv_rv_lore4")
TARGET_TASK="ImageNet100"
MODEL="ViT-B-32"


for config in "${MERGE_CONFIGS[@]}"; do
    python3 src/robust_vector_experiments.py \
        --merge-config configs/${config}.yaml \
        --model $MODEL \
        --batch-size 128 \
        --adversary-task $TARGET_TASK\
        --target-task $TARGET_TASK \
        --target-cls 1 \
        --patch-size 22 \
        --alpha 5 \
        --test-utility \
        --test-effectiveness True \
        --seed 1 \
        --save ./results/rv_experiments/${config}.csv
done

# python3 src/robust_vector_experiments.py \
#     --model ViT-B-32 \
#     --batch-size 128 \
#     --adversary-task 'CIFAR100' \
#     --target-task 'CIFAR100' \
# 	--target-cls 3 \
#     --patch-size 22 \
#     --alpha 5 \
#     --scaling-coef 0.2 \
#     --test-utility \
#     --test-effectiveness True \
#     --scale-weights

# CUDA_VISIBLE_DEVICES=0 python3 src/main_task_arithmetic_unified.py \
#     --model ViT-B-32 \
#     --model hf-hub:chs20/fare4-clip
#     --adversary-task 'CIFAR100' \
#     --target-task 'CIFAR100' \
# 	  --target-cls 1 \
#     --patch-size 22 \
#     --alpha 5 \
#     --scaling-coef 0.2 \
#     --test-utility \
#     --test-effectiveness True
