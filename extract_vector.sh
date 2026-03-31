# python3 src/extract_vector.py \
#     --model 'ViT-B-32' \
#     --vector-type 'fare'

# DATASETS=("CIFAR100" "ImageNet100" "Cars" "EuroSAT" "GTSRB" "PETS" "SUN397")
DATASETS=("ImageNet100")
MODE="backdoor"

# for i in {1..5}; do
for dataset in "${DATASETS[@]}"; do
    python3 src/extract_vector.py \
    --dataset $dataset\
    --model 'ViT-B-32' \
    --target-cls 1 \
    --seed 1 \
    --vector-type $MODE
done
# done

# for i in {1..3}; do
#     python3 src/extract_vector.py \
#         --dataset 'CIFAR100' \
#         --model 'ViT-B-32' \
#         --seed $i \
#         --vector-type $MODE
    
#     python3 src/extract_vector.py \
#         --dataset 'ImageNet100' \
#         --model 'ViT-B-32' \
#         --seed $i \
#         --vector-type $MODE
# done
