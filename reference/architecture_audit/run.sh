#!/bin/bash

# Ctrl+C 입력 시 모든 백그라운드 작업 종료
trap "echo 'Stopping...'; pkill -P $$; exit" SIGINT

# ################## 1번째 세트 ##################
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/cifar100_c/tent.yaml MODEL.ARCH Hendrycks2020AugMix_ResNeXt TEST.BATCH_SIZE 4 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg cfgs/cifar100_c/tent.yaml MODEL.ARCH Hendrycks2020AugMix_ResNeXt TEST.BATCH_SIZE 4 RNG_SEED 2 & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg cfgs/cifar100_c/tent.yaml MODEL.ARCH Hendrycks2020AugMix_ResNeXt TEST.BATCH_SIZE 4 RNG_SEED 3 & sleep 1.5

# wait      # 👉 여기서 위 세 개가 전부 끝날 때까지 기다림



# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/tent.yaml    MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 RNG_SEED 1 & sleep 1.5






# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/tent.yaml    MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/tent.yaml    MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 RNG_SEED 1 & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_r/source.yaml    MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_r/tent.yaml      MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_r/tent.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True with_bn True & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True with_bn True & sleep 1.5



# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_r/tent.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True with_bn True & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True with_bn True & sleep 1.5


# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 2 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 3 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 4 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 5 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 2 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 3 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5


# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar100_c/tent.yaml        TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 2               ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/cifar100_c/tent_gs.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 2 OPTIM.LR 0.01 ln True  lt True  ls True  with_bn False & sleep 1.5 # all
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/cifar100_c/tent_gs.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 2 OPTIM.LR 0.01 ln True  lt True  ls False with_bn False & sleep 1.5 # l only
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar100_c/tent_gs.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 2 OPTIM.LR 0.01 ln False lt False ls True  with_bn False & sleep 1.5 # c only



# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/tent.yaml      TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/tent_gs.yaml   TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01  ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/cmf.yaml       TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/roid.yaml      TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/cotta.yaml      TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/cifar10_c/tent_gs.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True  lt True  ls True  with_bn False & sleep 1.5 # all
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/cifar10_c/tent_gs.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True  lt True  ls False with_bn False & sleep 1.5 # l only
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar10_c/tent_gs.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln False lt False ls True  with_bn False & sleep 1.5 # c only



# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar10_c/tent.yaml      TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar10_c/tent.yaml      TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/cifar10_c/cotta.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5




# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/source_source.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar100_c/source_source.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/source_source.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5

# CUDA_VISIBLE_DEVICES=1 python test_time_source.py --cfg   cfgs/cifar100_c/tent_source.yaml         TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls False with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time_source.py --cfg   cfgs/cifar100_c/tent_gs_source.yaml      TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5


# CUDA_VISIBLE_DEVICES=2 python test_time_source.py --cfg   cfgs/cifar100_c/tent_gs_source.yaml      TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time_source.py --cfg   cfgs/cifar100_c/tent_gs_source.yaml      TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# 
# CUDA_VISIBLE_DEVICES=3 python test_time_source.py --cfg   cfgs/imagenet_c/tent_source.yaml       TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time_source.py --cfg   cfgs/imagenet_c/tent_gs_source.yaml    TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5



# CUDA_VISIBLE_DEVICES=1 python test_time_source.py --cfg   cfgs/cifar10_c/tent_source.yaml         TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls False with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time_source.py --cfg   cfgs/cifar10_c/tent_gs_source.yaml        TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time_source.py --cfg   cfgs/cifar10_c/tent_gs.yaml      TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls False with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar10_c_resnext/tent.yaml         TEST.BATCH_SIZE 16 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/cifar10_c_resnext/tent_gs.yaml      TEST.BATCH_SIZE 16 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5


# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar100_c/tent.yaml         TEST.BATCH_SIZE 16 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/cifar100_c/tent_gs.yaml      TEST.BATCH_SIZE 16 RNG_SEED 1 ADAP_RANGE 1 OPTIM.LR 0.01 ln True lt True ls True with_bn False & sleep 1.5



# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/source.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 & sleep 1.5

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/eta.yaml        TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-3 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar10_c/tent.yaml        TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-3 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
CUDA_VISIBLE_DEVICES=7 python test_time.py --cfg   cfgs/cifar10_c/tent_pppp.yaml   TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn True & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/cifar10_c/eta_pppp.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/cifar10_c/deyo_pppp.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar10_c/tent_acon.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/cifar10_c/eta_acon.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/cifar10_c/deyo_acon.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5


# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar10_c/tent_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/cifar10_c/eta_pau.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/cifar10_c/deyo_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar10_c/tent_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/cifar10_c/eta_pau.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/cifar10_c/deyo_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar100_c/source.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar100_c/tent.yaml      TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/cifar100_c/eta.yaml       TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/cifar100_c/deyo.yaml      TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar100_c/tent_gs.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar100_c/eta_gs.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/cifar100_c/deyo_gs.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar100_c/tent_pppp.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/cifar100_c/eta_pppp.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/cifar100_c/deyo_pppp.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar100_c/tent_acon.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/cifar100_c/eta_acon.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/cifar100_c/deyo_acon.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar100_c/tent_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/cifar100_c/eta_pau.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/cifar100_c/deyo_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5



# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c/source.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/tent.yaml       TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/eta.yaml        TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c/deyo.yaml       TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c/tent_gs.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 0.005 ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_c/eta_gs.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 0.005 ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/deyo_gs.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 0.005 ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5


# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/tent_pppp.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 0.005 ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/eta_pppp.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 0.005 ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c/deyo_pppp.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 0.005 ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_c/tent_acon.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 0.005 ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/eta_acon.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 0.005 ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/imagenet_c/deyo_acon.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 0.005 ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/tent_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/eta_pau.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c/deyo_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_c/tent_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/eta_pau.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/imagenet_c/deyo_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 4 with_bn False & sleep 1.5

# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar100_c/tent_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/cifar100_c/eta_pau.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/cifar100_c/deyo_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1  ln True lt True ls True ADAP_RANGE 2 with_bn False & sleep 1.5






# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar10_c/tent_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/cifar10_c/eta_pau.yaml     TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/cifar10_c/deyo_pau.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 OPTIM.LR 1e-2 ln True lt True ls True ADAP_RANGE 1 with_bn False & sleep 1.5




# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar100_c/cotta.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/cotta.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/cotta_gs.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar100_c/cotta_gs.yaml    TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml    TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/cotta.yaml       TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar100_c/cotta.yaml      TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/cotta.yaml      TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/cotta_gs.yaml    TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 5  ln True lt True ls False  & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar100_c/cotta_gs.yaml   TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 5  ln True lt True ls False  & sleep 1.5
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml   TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 2  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml   TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 3  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml   TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 4  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml   TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 5  ln True lt True ls True  & sleep 1.5



# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/cifar10_c/cotta_gs.yaml    TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 5  ln True lt True ls False  & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar100_c/cotta_gs.yaml   TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 5  ln True lt True ls False  & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml   TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 2 ln True lt True ls True lb True & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml   TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True lb True & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml   TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 2  ln True lt True ls True lb True & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml   TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 3  ln True lt True ls True lb True & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_c/cotta_gs.yaml   TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 4  ln True lt True ls True lb True & sleep 1.5




################################################################################################################################################################

# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta.yaml      TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta.yaml      TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta.yaml      TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta_vit.yaml      TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c_vit/source.yaml       TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta_vit.yaml    TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c_vit/source.yaml         TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c/tent_full.yaml           TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/tent_full_gs.yaml        TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/tent_full_gs.yaml        TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 2  ln True lt True ls True  & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c/tent_full_gs.yaml     TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/tent_full_gs.yaml     TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 2  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/tent_full_gs.yaml     TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 3  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/tent_full_gs.yaml     TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 4  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c/tent_full_gs.yaml     TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5

# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c_vit/tent_full.yaml        TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True lb True & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c_vit/tent_full_gs.yaml     TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 0  ln True lt True ls True lb True & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c_vit/tent_full_gs.yaml     TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True lb True & sleep 1.5

# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c_vit/tent_full.yaml        TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True lb True & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_c_vit/tent_full_gs.yaml     TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 0  ln True lt True ls True lb True & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c_vit/tent_full_gs.yaml     TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True lb True & sleep 1.5


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c/memo.yaml        RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/imagenet_c/memo_gs.yaml     RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c/memo_gs.yaml     RNG_SEED 1 ADAP_RANGE 2  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c/memo_gs.yaml     RNG_SEED 1 ADAP_RANGE 3  ln True lt True ls True  & sleep 1.5






# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c/memo.yaml        TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_c/memo_gs.yaml     TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/memo_gs.yaml     TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 2  ln True lt True ls True  & sleep 1.5



# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta_vit.yaml       TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta_vit_gs.yaml    TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 2  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta_vit_gs.yaml   TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 4  ln True lt True ls True  & sleep 1.5



# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_c/tent.yaml           TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/tent_gs.yaml        TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/tent_full.yaml      TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/tent_full_gs.yaml   TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 2  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=7 python test_time.py --cfg   cfgs/imagenet_c/tent_full_gs.yaml   TEST.BATCH_SIZE 128  RNG_SEED 1 ADAP_RANGE 4  ln True lt True ls True  & sleep 1.5


# 
# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta_gs.yaml   TEST.BATCH_SIZE 64  RNG_SEED 1 ADAP_RANGE 5  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_c_vit/cotta_gs.yaml   TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 5  ln True lt True ls True  & sleep 1.5

################################################################################################################################################################


# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg    cfgs/imagenet_c_vit/tent_gs.yaml   MODEL.ARCH vit_base_patch16_224.orig_in21k_ft_in1k   OPTIM.LR  2.5e-4  TEST.BATCH_SIZE 128 RNG_SEED 1 ADAP_RANGE 1 ln True  lt True  ls True with_bn True & sleep 1.5

### do transformer!@!!!!!

# CUDA_VISIBLE_DEVICES=1 python test_time.py --cfg   cfgs/cifar10_c/memo_gs.yaml      RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/cifar100_c/memo_gs.yaml     RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/imagenet_c/memo_gs.yaml     RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5



# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar100_c/source.yaml     TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_c/source.yaml    TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls True  & sleep 1.5


# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/cifar100_c/memo.yaml                           RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=7 python test_time.py --cfg   cfgs/imagenet_c/memo.yaml                           RNG_SEED 1 & sleep 1.5

# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/cifar10_c/source.yaml     MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/cifar10_c/source.yaml     MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5









# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar10_c/source.yaml     MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/cifar100_c/source.yaml    MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg   cfgs/imagenet_c/source.yaml    MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 & sleep 1.5


# CUDA_VISIBLE_DEVICES=2 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt False ls False with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln False lt True ls False with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=4 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt True ls False with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=5 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln True lt False ls True with_bn False & sleep 1.5
# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/imagenet_r/tent_gs.yaml   MODEL.ARCH resnet50  TEST.BATCH_SIZE 64 RNG_SEED 1 ADAP_RANGE 1  ln False lt True ls True with_bn False & sleep 1.5

# # 

# CUDA_VISIBLE_DEVICES=6 python test_time.py --cfg   cfgs/imagenet_c/source.yaml  MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 RNG_SEED 1 & sleep 1.5
# CUDA_VISIBLE_DEVICES=7 python test_time.py --cfg   cfgs/imagenet_c/tent.yaml    MODEL.ARCH resnet50  TEST.BATCH_SIZE 128 RNG_SEED 1 & sleep 1.5




wait      # 마지막 세트까지 끝나면 스크립트 종료

