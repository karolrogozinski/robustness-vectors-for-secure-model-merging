DATASETS=("CIFAR100" "ImageNet100" "Cars" "EuroSAT" "GTSRB" "PETS" "SUN397")

for i in {1..5}; do
    for dataset in "${DATASETS[@]}"; do
        echo "********************************************" 
        echo $dataset $i
        echo "********************************************" 

        CUDA_VISIBLE_DEVICES=1 python3 src/ut_badmergingon.py --adversary-task $dataset --model "ViT-L-14" --target-cls $i --mask-length 22 --seed 1
        CUDA_VISIBLE_DEVICES=1 python3 src/finetune_backdoor_badmergingon.py --adversary-task $dataset --model "ViT-L-14" --target-cls $i --patch-size 22 --alpha 5 --seed 1
    done
done

# DATASETS=("ImageNet100" "Cars" "EuroSAT" "GTSRB" )
# DATASETS=("GTSRB")

# for i in {1..3}; do
#     for dataset in "${DATASETS[@]}"; do
#         echo "********************************************" 
#         echo $dataset $i
#         echo "********************************************" 

#         # CUDA_VISIBLE_DEVICES=2 python3 src/ut_badmergingon.py --adversary-task $dataset --model "ViT-B-32" --target-cls 1 --mask-length 22 --seed $i
#         CUDA_VISIBLE_DEVICES=2 python3 src/finetune_backdoor_badmergingon.py --adversary-task $dataset --model "ViT-B-32" --target-cls 1 --patch-size 22 --alpha 5 --seed $i
#     done
# done

