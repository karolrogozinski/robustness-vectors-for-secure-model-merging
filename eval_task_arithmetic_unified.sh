# for i in {1..3}; do
python3 src/main_task_arithmetic_unified.py \
    --merge-config configs/cifar100_st.yaml \
    --model ViT-B-32 \
    --batch-size 128 \
    --adversary-task 'ImageNet100' \
    --target-task 'ImageNet100' \
    --target-cls 1 \
    --patch-size 22 \
    --alpha 5 \
    --scaling-coef- 0.2 \
    --test-utility \
    --test-effectiveness True \
    --seed 1 \
    --save ./results/test.csv
        # --save ./results/table_2/single_clean.csv
# done

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
