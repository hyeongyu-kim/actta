"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""
import torch
import torch.nn as nn

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, SymmetricCrossEntropy, SoftLikelihoodRatio


@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x



@ADAPTATION_REGISTRY.register()
class Tent_gs(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        self.with_bn = cfg.with_bn

        super().__init__(cfg, model, num_classes)

        self.with_bn = cfg.with_bn
        # setup loss function
        self.softmax_entropy = Entropy()
        self.batch_size = cfg.TEST.BATCH_SIZE


        
        # self.gs_reg_lambda = 1e-2
        # self.gs_init = {}
        # for name, p in self.model.named_parameters():
        #     if "gsL" in name:
        #         self.gs_init[name] = p.detach().clone()



    # def gs_reg_loss(self):

    #     if self.gs_reg_lambda <= 0.0:
    #         return 0.0
    #     reg = 0.0
    #     for name, p in self.model.named_parameters():
    #         if "gsL" in name:
    #             # 초기값이 없는 경우(새로 생긴 파라미터 등)는 스킵
    #             if name not in self.gs_init:
    #                 continue
    #             reg = reg + (p - self.gs_init[name]).pow(2).sum()
    #     # scale by lambda
    #     return self.gs_reg_lambda * reg


    def loss_calculation(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        loss = self.softmax_entropy(outputs).mean(0)
        

        return outputs, loss





    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []

        if self.with_bn : 
            for nm, m in self.model.named_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{np}")



        for np, p in self.model.named_parameters():
            print(np)
            if 'gsL' in np:
                params.append(p); names.append(np)

        print(names,'names!!!')

        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics

        ################################## Remain un touched ################################
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
        ################################## Remain un touched ################################


        ################################## BN affine grad ################################
        if self.with_bn : 
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                elif isinstance(m, nn.BatchNorm1d):
                    m.requires_grad_(True)
                elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                    m.requires_grad_(True)

        ################################## BN affine grad ################################
        for name, p in self.model.named_parameters():
            if 'gsL' in name:
                p.requires_grad_(True)
            