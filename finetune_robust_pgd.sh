# python3 src/finetune_robust_pgd.py --model "hf-hub:chs20/fare4-clip"

DATASETS=("CIFAR100" "ImageNet100" "Cars" "EuroSAT" "GTSRB" "PETS" "SUN397")

for i in {1..5}; do
    for dataset in "${DATASETS[@]}"; do
        echo "********************************************" 
        echo $dataset $i
        echo "********************************************" 

        CUDA_VISIBLE_DEVICES=1 python3 src/finetune_robust_pgd.py --model ViT-L-14 --dataset $dataset --seed $i
    done
done
