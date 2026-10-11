import os
import torch
import logging
import numpy as np
import methods
import math
import shutil

from models.model import get_model
from models.custom_standard import WideResNetTTA, load_partial_weights_from_model, load_partial_weights_from_model_exclude_prefix
from models.custom_standardv2 import WideResNetTTAv2
from models.custom_standardv12 import WideResNetTTAv12
from models.custom_standard_cifar10_resnext import CifarResNeXt_TTA
from models.custom_standard_cifar100 import Hendrycks2020AugMixResNeXtNet
from models.custom_standard_cifar100_wrn40 import WideResNet_TTA
from models.torchvision.vision.torchvision.models.vision_transformer_custom import vit_b_16, ViT_B_16_Weights
from models.torchvision.vision.torchvision.models.resnet_custom import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

import timm

import pickle

from timm.models import resnetv2_50x1_bit
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from copy import deepcopy
from torchvision import transforms
from robustbench.model_zoo.architectures.utils_architectures import normalize_model

from utils.misc import print_memory_info
from utils.eval_utils import get_accuracy, eval_domain_dict
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader, get_source_loader
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq
from torchvision.transforms import InterpolationMode


logger = logging.getLogger(__name__)
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################



import os
import time
import threading
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo
)

class VRAMMonitor:
    def __init__(self, device_index=0, interval=0.01):
        self.device_index = device_index
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.baseline = 0
        self.peak = 0
        self.samples = []

        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(device_index)

    def _used_bytes(self):
        info = nvmlDeviceGetMemoryInfo(self.handle)
        return info.used  # nvidia-smi memory.used와 거의 같은 값

    def _loop(self):
        while not self._stop.is_set():
            cur = self._used_bytes()
            self.samples.append(cur)
            if cur > self.peak:
                self.peak = cur
            time.sleep(self.interval)

    def start(self):
        self.baseline = self._used_bytes()
        self.peak = self.baseline
        self.samples = [self.baseline]
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        final = self._used_bytes()
        if final > self.peak:
            self.peak = final
        return {
            "baseline_mib": self.baseline / (1024 ** 2),
            "peak_mib": self.peak / (1024 ** 2),
            "delta_peak_mib": (self.peak - self.baseline) / (1024 ** 2),
            "final_mib": final / (1024 ** 2),
        }

    def close(self):
        try:
            nvmlShutdown()
        except Exception:
            pass
















####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################





from models.rensext import CifarResNeXt, ResNeXtBottleneck

            
class Hendrycks2020AugMixResNeXtNet_learnrelu_cifar100(CifarResNeXt):
    def __init__(self, depth=29, cardinality=4, base_width=32):
        super().__init__(ResNeXtBottleneck,
                        depth=depth,
                        num_classes=100,
                        cardinality=cardinality,
                        base_width=base_width)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)

class Hendrycks2020AugMixResNeXtNet_learnrelu_cifar10(CifarResNeXt):
    def __init__(self, depth=29, cardinality=4, base_width=32):
        super().__init__(ResNeXtBottleneck,
                        depth=depth,
                        num_classes=10,
                        cardinality=cardinality,
                        base_width=base_width)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)

        
############################################################################################################################################



def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated"
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)



    base_model, model_preprocess = get_model(cfg, num_classes, device)

    print(model_preprocess, 'model_preprocess')
    # print(base_model, 'base_model')
    # print(model_preprocess, 'model_preprocess')
    # base_model.preprocess = None
    

    if 'gs' in cfg.MODEL.ADAPTATION:
        if  'vit' in cfg.MODEL.ARCH :
            from models.gs_model import vit_b_16_gs2
            base_model = vit_b_16_gs2(model=base_model, adap_layer=cfg.ADAP_RANGE , ln = cfg.ln,  lt = cfg.lt,  ls = cfg.ls , lb= cfg.lb)
            base_model.to(device)

        elif  'resnet50' in cfg.MODEL.ARCH :
            # print('modifying resnet50 for gs...')
            # print('modifying resnet50 for gs...')
            # print('modifying resnet50 for gs...')

            from models.gs_model import resnet50_gs2
            base_model = resnet50_gs2(model=base_model, adap_layer=cfg.ADAP_RANGE , ln = cfg.ln,  lt = cfg.lt,  ls = cfg.ls , lb= cfg.lb)
            base_model.to(device)
        

        elif 'ResNeXt' in cfg.MODEL.ARCH and  cfg.CORRUPTION.DATASET == 'cifar100_c':

            from models.gs_model import resnext_gs2
            new_model = Hendrycks2020AugMixResNeXtNet_learnrelu_cifar100().cuda() # GPU 사용 시
            if hasattr(base_model, 'module'):
                old_state_dict = base_model.module.state_dict()
            else:
                old_state_dict = base_model.state_dict()
            new_state_dict = {}
            for k, v in old_state_dict.items():
                new_k = k.replace('module.', '') # DataParallel 접두사 제거
                new_state_dict[new_k] = v

            msg = new_model.load_state_dict(new_state_dict, strict=False)
            print(f"[Weight Load Info] Missing: {msg.missing_keys}")

            base_model = resnext_gs2(model=new_model, adap_layer=cfg.ADAP_RANGE , ln = cfg.ln,  lt = cfg.lt,  ls = cfg.ls , lb= cfg.lb)
            base_model.to(device)
    
    
        elif 'ResNeXt' in cfg.MODEL.ARCH and  cfg.CORRUPTION.DATASET == 'cifar10_c':

            from models.gs_model import resnext_gs2
            new_model = Hendrycks2020AugMixResNeXtNet_learnrelu_cifar10().cuda() # GPU 사용 시
            if hasattr(base_model, 'module'):
                old_state_dict = base_model.module.state_dict()
            else:
                old_state_dict = base_model.state_dict()
            new_state_dict = {}
            for k, v in old_state_dict.items():
                new_k = k.replace('module.', '') # DataParallel 접두사 제거
                new_state_dict[new_k] = v

            msg = new_model.load_state_dict(new_state_dict, strict=False)
            print(f"[Weight Load Info] Missing: {msg.missing_keys}")
            base_model = resnext_gs2(model=new_model, adap_layer=cfg.ADAP_RANGE , ln = cfg.ln,  lt = cfg.lt,  ls = cfg.ls , lb= cfg.lb)
            base_model.to(device)
    


        else :
            from models.gs_model import cnn_gs2
            base_model = cnn_gs2(model=base_model, adap_layer=cfg.ADAP_RANGE, ln = cfg.ln,  lt = cfg.lt,  ls = cfg.ls , lb= cfg.lb)
            base_model.to(device)





    if 'pppp' in cfg.MODEL.ADAPTATION:
        if  'vit' in cfg.MODEL.ARCH :
            from models.p_model import vit_b_16_p
            base_model = vit_b_16_p(model=base_model, adap_layer=cfg.ADAP_RANGE)
            base_model.to(device)

        elif  'resnet50' in cfg.MODEL.ARCH :
            from models.p_model import resnet50_cnn_p
            base_model = resnet50_cnn_p(model=base_model, adap_layer=cfg.ADAP_RANGE )
            base_model.to(device)
        
        elif 'ResNeXt' in cfg.MODEL.ARCH and  cfg.CORRUPTION.DATASET == 'cifar100_c':
            from models.p_model import resnext_cnn_p
            new_model = Hendrycks2020AugMixResNeXtNet_learnrelu_cifar100().cuda() # GPU 사용 시
            if hasattr(base_model, 'module'):
                old_state_dict = base_model.module.state_dict()
            else:
                old_state_dict = base_model.state_dict()
            new_state_dict = {}
            for k, v in old_state_dict.items():
                new_k = k.replace('module.', '') # DataParallel 접두사 제거
                new_state_dict[new_k] = v

            msg = new_model.load_state_dict(new_state_dict, strict=False)
            print(f"[Weight Load Info] Missing: {msg.missing_keys}")

            base_model = resnext_cnn_p(model=new_model, adap_layer=cfg.ADAP_RANGE )
            base_model.to(device)

        else :
            from models.p_model import cnn_p
            base_model = cnn_p(model=base_model, adap_layer=cfg.ADAP_RANGE)
            base_model.to(device)



    if 'acon' in cfg.MODEL.ADAPTATION:
        if  'vit' in cfg.MODEL.ARCH :
            from models.acon_model import vit_b_16_acon
            base_model = vit_b_16_acon(model=base_model, adap_layer=cfg.ADAP_RANGE)
            base_model.to(device)


        elif  'resnet50' in cfg.MODEL.ARCH :
            from models.acon_model import resnet50_cnn_acon
            base_model = resnet50_cnn_acon(model=base_model, adap_layer=cfg.ADAP_RANGE )
            base_model.to(device)
        
        elif 'ResNeXt' in cfg.MODEL.ARCH and  cfg.CORRUPTION.DATASET == 'cifar100_c':
            
            from models.acon_model import resnesxt_cnn_acon
            new_model = Hendrycks2020AugMixResNeXtNet_learnrelu_cifar100().cuda() # GPU 사용 시
            if hasattr(base_model, 'module'):
                old_state_dict = base_model.module.state_dict()
            else:
                old_state_dict = base_model.state_dict()
            new_state_dict = {}
            for k, v in old_state_dict.items():
                new_k = k.replace('module.', '') # DataParallel 접두사 제거
                new_state_dict[new_k] = v

            msg = new_model.load_state_dict(new_state_dict, strict=False)
            print(f"[Weight Load Info] Missing: {msg.missing_keys}")

            base_model = resnesxt_cnn_acon(model=new_model, adap_layer=cfg.ADAP_RANGE )
            base_model.to(device)



        else :
            from models.acon_model import cnn_acon
            base_model = cnn_acon(model=base_model, adap_layer=cfg.ADAP_RANGE)
            base_model.to(device)
            

    if 'pau' in cfg.MODEL.ADAPTATION:
        if  'vit' in cfg.MODEL.ARCH :
            from models.pau_model import vit_b_16_pau
            base_model = vit_b_16_pau(model=base_model, adap_layer=cfg.ADAP_RANGE)
            base_model.to(device)


        elif  'resnet50' in cfg.MODEL.ARCH :
            from models.pau_model import resnet50_cnn_pau
            base_model = resnet50_cnn_pau(model=base_model, adap_layer=cfg.ADAP_RANGE )
            base_model.to(device)

        elif 'ResNeXt' in cfg.MODEL.ARCH and  cfg.CORRUPTION.DATASET == 'cifar100_c':
        
            from models.pau_model import resnext_cnn_pau
            new_model = Hendrycks2020AugMixResNeXtNet_learnrelu_cifar100().cuda() # GPU 사용 시
            if hasattr(base_model, 'module'):
                old_state_dict = base_model.module.state_dict()
            else:
                old_state_dict = base_model.state_dict()
            new_state_dict = {}
            for k, v in old_state_dict.items():
                new_k = k.replace('module.', '') # DataParallel 접두사 제거
                new_state_dict[new_k] = v

            msg = new_model.load_state_dict(new_state_dict, strict=False)
            print(f"[Weight Load Info] Missing: {msg.missing_keys}")

            base_model = resnext_cnn_pau(model=new_model, adap_layer=cfg.ADAP_RANGE )
            base_model.to(device)


        else :
            from models.pau_model import cnn_pau
            base_model = cnn_pau(model=base_model, adap_layer=cfg.ADAP_RANGE)
            base_model.to(device)
            






    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET == "domainnet126":
        # extract the domain sequence for a specific checkpoint.
        domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
        # domain_sequence = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        domain_sequence = cfg.CORRUPTION.TYPE
    logger.info(f"Using {cfg.CORRUPTION.DATASET} with the following domain sequence: {domain_sequence}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # setup the severities for the gradual setting
    # if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
    #     severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    #     logger.info(f"Using the following severity sequence for each domain: {severities}")
    # else:
    severities = cfg.CORRUPTION.SEVERITY




    errs = []
    errs_5 = []
    domain_dict = {}

    print(domain_seq_loop,'domain_seq_loop')

    # start evaluation
    for i_dom, domain_name in enumerate(domain_seq_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except AttributeError:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")


        # print(cfg.CORRUPTION.NUM_EX,'cfg.CORRUPTION.NUM_EX')
        # print(cfg.CORRUPTION.NUM_EX,'cfg.CORRUPTION.NUM_EX')
        # print(cfg.CORRUPTION.NUM_EX,'cfg.CORRUPTION.NUM_EX')
        # print(cfg.CORRUPTION.NUM_EX,'cfg.CORRUPTION.NUM_EX')


        for severity in severities:
            test_data_loader = get_test_loader(
                setting=cfg.SETTING,
                adaptation=cfg.MODEL.ADAPTATION,
                dataset_name=cfg.CORRUPTION.DATASET,
                preprocess=model_preprocess,
                data_root_dir=cfg.DATA_DIR,
                domain_name=domain_name,
                domain_names_all=domain_sequence,
                severity=severity,
                num_examples=cfg.CORRUPTION.NUM_EX,
                rng_seed=cfg.RNG_SEED,
                use_clip=cfg.MODEL.USE_CLIP,
                n_views=cfg.TEST.N_AUGMENTATIONS,
                delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=True,
                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
            )

            if i_dom == 0:
                # Note that the input normalization is done inside of the model
                logger.info(f"Using the following data transformation:\n{test_data_loader.dataset.transform}")

            # evaluate the model
            # log_gpu_memory("Before inference")

            # source_dataset, source_data_loader = get_source_loader(
            #         dataset_name='cifar100', 
            #         adaptation=cfg.MODEL.ADAPTATION, 
            #         preprocess=model_preprocess,
            #         data_root_dir=cfg.DATA_DIR, 
            #         batch_size=1, 
            #         use_clip=cfg.MODEL.USE_CLIP, 
            #         n_views=cfg.TEST.N_AUGMENTATIONS,
            #         train_split=True, 
            #         ckpt_path=cfg.MODEL.CKPT_PATH, 
            #         num_samples=1000,
            #         percentage=1.0, 
            #         workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
            # )

            # sources_all = {'imgs': [], 'labels': []}




            monitor = VRAMMonitor(device_index=0, interval=0.01)  # 10ms polling
            monitor.start()
            acc, domain_dict, num_samples = get_accuracy(
                model,
                data_loader=test_data_loader,
                dataset_name=cfg.CORRUPTION.DATASET,
                domain_name=domain_name,
                setting=cfg.SETTING,
                domain_dict=domain_dict,
                print_every=cfg.PRINT_EVERY,
                device=device)
                
            vram_stats = monitor.stop()
            monitor.close()

            logger.info(
                f"[VRAM] baseline={vram_stats['baseline_mib']:.1f} MiB | "
                f"peak={vram_stats['peak_mib']:.1f} MiB | "
                f"delta_peak={vram_stats['delta_peak_mib']:.1f} MiB | "
                f"final={vram_stats['final_mib']:.1f} MiB"
            )
            # print(acc,'acc')
            # log_gpu_memory("After inference")

            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}")

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")

    # if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
    #     # print detailed results for each domain
    #     eval_domain_dict(domain_dict, domain_seq=domain_sequence)

    if cfg.TEST.DEBUG:
        print_memory_info()


if __name__ == '__main__':

    evaluate('"Evaluation.')
