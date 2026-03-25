for i in {1..3}; do
    python3 src/robust_vector_experiments_rv.py \
        --model ViT-B-32 \
        --batch-size 128 \
        --adversary-task 'CIFAR100' \
        --target-task 'CIFAR100' \
        --target-cls $i \
        --patch-size 22 \
        --alpha 5 \
        --scaling-coef- 0.2 \
        --test-utility \
        --test-effectiveness True \
        --seed 1 \
        --save ./results/figure_2/st_rv_cifar100.csv
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
