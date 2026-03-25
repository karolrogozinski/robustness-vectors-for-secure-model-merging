CUDA_VISIBLE_DEVICES=0 python3 src/main_task_arithmetic_robust_pgd.py \
    --model ViT-B-32 \
    --attack-type 'Clean' \
    --adversary-task 'CIFAR100' \
    --scaling-coef 0.3 \
    --test-utility \
    --test-effectiveness True

CUDA_VISIBLE_DEVICES=0 python3 src/main_task_arithmetic_robust_pgd.py \
    --model ViT-B-32 \
    --attack-type 'On' \
    --adversary-task 'CIFAR100' \
    --scaling-coef 0.3 \
    --test-utility \
    --test-effectiveness True