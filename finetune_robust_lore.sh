# python3 src/finetune_robust_pgd.py --model "hf-hub:chs20/fare4-clip"

DATASETS=("ImageNet")
GPUS="1,2"

# for i in {1..5}; do
    for dataset in "${DATASETS[@]}"; do
        echo "********************************************" 
        # echo $dataset $i
        echo "********************************************" 

        CUDA_VISIBLE_DEVICES=$GPUS python3 src/finetune_robust_lore.py --model ViT-B-32 --dataset $dataset --seed 1
    done
# done
