import os
import torch
import logging
import numpy as np
import methods
import math
import shutil

from models.model import get_model
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
# GroupNorm 지원하는 get_norm
import torch.nn as nn
def get_norm(norm, num_channels):
    if norm == "GN":
        return nn.GroupNorm(32, num_channels)
    elif norm == "BN":
        return nn.BatchNorm2d(num_channels)
    elif norm is None:
        return None
    else:
        raise ValueError(f"Unsupported norm: {norm}")

# Conv2d 래퍼
class Conv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, norm=None, **kwargs):
        layers = []
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs)
        layers.append(conv)
        if norm is not None:
            layers.append(norm)
        super().__init__(*layers)
        self.conv = conv  # for init

# Detectron2-style weight init
def c2_msra_fill(module):
    if isinstance(module, Conv2d):
        nn.init.kaiming_normal_(module.conv.weight, mode="fan_out", nonlinearity="relu")

############################################################################################################################################

def load_partial_weights_from_model_exclude_prefix(target_model, source_model):
    src_sd, tgt_sd = source_model.state_dict(), target_model.state_dict()

    has_prefix_src = any(k.startswith('model.') for k in src_sd)
    has_prefix_tgt = any(k.startswith('model.') for k in tgt_sd)

    def strip(k):
        return k[len('model.'):] if k.startswith('model.') else k

    def add(k):
        return 'model.' + k if not k.startswith('model.') else k

    mapped = {}
    for k, v in src_sd.items():
        # ① 둘 다 prefix → 그대로 둔다
        if has_prefix_src and has_prefix_tgt:
            new_k = k
        # ② source 만 prefix → 잘라 낸다
        elif has_prefix_src and not has_prefix_tgt:
            new_k = strip(k)
        # ③ target 만 prefix → prefix를 붙인다
        elif not has_prefix_src and has_prefix_tgt:
            new_k = add(k)
        else:
            new_k = k

        if new_k in tgt_sd and tgt_sd[new_k].shape == v.shape:
            mapped[new_k] = v

    # 로드
    tgt_sd.update(mapped)
    msg = target_model.load_state_dict(tgt_sd, strict=False)

    print(f"[load_partial_weights] loaded {len(mapped)} / {len(tgt_sd)} tensors")
    print("missing:", len(msg.missing_keys), "unexpected:", len(msg.unexpected_keys))
################################################################################################

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
    # for name, param in base_model.named_parameters():
    #     print(name)
    # exit()
    
    if cfg.MODEL.ARCH == 'resnet50':
        from models.custom_standard_act import convert_resnet50_with_custom_block
        base_model = convert_resnet50_with_custom_block(model=base_model)
        print("convert_resnet50_with_custom_block success")
        
    # print("base_model", base_model)
    # exit()
        

    if 'gs' in cfg.MODEL.ADAPTATION:
        if  'vit' in cfg.MODEL.ARCH :
            from models.custom_standard_act import vit_b_16_gs2
            base_model = vit_b_16_gs2(model=base_model, adap_layer=cfg.ADAP_RANGE , ln = cfg.ln,  lt = cfg.lt,  ls = cfg.ls, consis_weight = 0.0 )
            base_model.to(device)
        else :
            from models.custom_standard_act import model_gs2_by_count
            base_model = model_gs2_by_count(model=base_model, num_replacements_total=cfg.ADAP_RANGE, start_point=0,
                                            ln=cfg.ln, lt=cfg.lt, ls=cfg.ls, consis_weight=0.0, preset='relu')
            base_model.to(device)
            

    
    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    if 'gs' in cfg.MODEL.ADAPTATION:
        model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    else:
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
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY


    
    errs = []
    errs_5 = []
    domain_dict = {}

    print(domain_seq_loop,'domain_seq_loop')
    
    _, source_data_loader = get_source_loader(
        dataset_name=cfg.CORRUPTION.DATASET,
        adaptation=cfg.MODEL.ADAPTATION,
        preprocess=model_preprocess,
        data_root_dir=cfg.DATA_DIR,
        batch_size=cfg.TEST.BATCH_SIZE,
        use_clip=cfg.MODEL.USE_CLIP,
        n_views=cfg.TEST.N_AUGMENTATIONS,
    )

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
                shuffle=False,
                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
            )

            if i_dom == 0:
                # Note that the input normalization is done inside of the model
                logger.info(f"Using the following data transformation:\n{test_data_loader.dataset.transform}")



            acc, domain_dict, num_samples = get_accuracy(
                model,
                data_loader=test_data_loader,
                dataset_name=cfg.CORRUPTION.DATASET,
                domain_name=domain_name,
                setting=cfg.SETTING,
                domain_dict=domain_dict,
                print_every=cfg.PRINT_EVERY,
                device=device)
            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)
            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}")
            
            
            # acc, domain_dict, num_samples = get_accuracy(
            #     model,
            #     data_loader=source_data_loader,
            #     dataset_name=cfg.CORRUPTION.DATASET,
            #     domain_name=domain_name,
            #     setting=cfg.SETTING,
            #     domain_dict=domain_dict,
            #     print_every=cfg.PRINT_EVERY,
            #     device=device)            
            # err = 1. - acc
            # errs.append(err)
            # if severity == 5 and domain_name != "none":
            #     errs_5.append(err)
            # logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}")
            # exit()
            
            

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")

    if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=domain_sequence)

    if cfg.TEST.DEBUG:
        print_memory_info()


if __name__ == '__main__':

    evaluate('"Evaluation.')


