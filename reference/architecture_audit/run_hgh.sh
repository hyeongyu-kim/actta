#!/bin/bash

# Ctrl+C 입력 시 모든 백그라운드 작업 종료
trap "echo 'Stopping...'; pkill -P $$; exit" SIGINT




# CUDA_VISIBLE_DEVICES=0 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 with_bn False ADAP_RANGE 5  OPTIM.LR 0.0025 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 with_bn False ADAP_RANGE 13 OPTIM.LR 0.0025 ln True lt True ls True with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=0 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 with_bn False ADAP_RANGE 25 OPTIM.LR 0.0025 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 with_bn False ADAP_RANGE 38 OPTIM.LR 0.0025 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 with_bn False ADAP_RANGE 49 OPTIM.LR 0.0025 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg cfgs/imagenet_c_vit/tent_gs.yaml  MODEL.ARCH vit_base_patch16_224.orig_in21k_ft_in1k  TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 12 ln True lt True ls True with_bn False &


# CUDA_VISIBLE_DEVICES=0 python test_time_hgh.py --cfg cfgs/imagenet_c_vit/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH vit_base_patch16_224.orig_in21k_ft_in1k  TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time_hgh.py --cfg cfgs/imagenet_c_vit/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH vit_base_patch16_224.orig_in21k_ft_in1k  TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 3 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time_hgh.py --cfg cfgs/imagenet_c_vit/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH vit_base_patch16_224.orig_in21k_ft_in1k  TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 6 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time_hgh.py --cfg cfgs/imagenet_c_vit/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH vit_base_patch16_224.orig_in21k_ft_in1k  TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 9 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time_hgh.py --cfg cfgs/imagenet_c_vit/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH vit_base_patch16_224.orig_in21k_ft_in1k  TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 12 ln True lt True ls True with_bn False & sleep 1.5




# CUDA_VISIBLE_DEVICES=0 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH resnet50  TEST.BATCH_SIZE 128  ADAP_RANGE 5   ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH resnet50  TEST.BATCH_SIZE 128  ADAP_RANGE 13  ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH resnet50  TEST.BATCH_SIZE 128  ADAP_RANGE 25  ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH resnet50  TEST.BATCH_SIZE 128  ADAP_RANGE 38  ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time_hgh.py --cfg cfgs/imagenet_c/tent_gs.yaml OPTIM.LR 0.0025 MODEL.ARCH resnet50  TEST.BATCH_SIZE 128  ADAP_RANGE 49  ln True lt True ls True with_bn False & sleep 1.5


# wait