from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# =========================
# 1) GateSpline-2 (GS2)
# =========================
import math


import re

class GS2(nn.Module):
    """
    GateSpline-2 (shift + gated-slope residual on top of base activation)

    g(x) = base(x - shift) + lambda(x) * (x - shift)

    where
      base : ReLU or GELU(tanh approx) depending on preset
      lambda(x) = neg + (pos - neg) * sigmoid(beta * (x - shift))

    - preset: 'relu' | 'gelu' | None
      * 'relu' : base = ReLU, 초기값은 정확히 ReLU
      * 'gelu' : base = GELU(tanh approx), 초기값은 정확히 GELU
      * None   : base = identity (그냥 x)

    - share: 'channel' | 'layer'
      * 'channel' : 채널마다 파라미터(길이 = channels)
      * 'layer'   : 레이어 전체에 스칼라 파라미터 1개

    - learn_neg / learn_pos / learn_shift:
      * True  : nn.Parameter 로 만들고 이름에 *_gs 붙임 (학습됨)
      * False : buffer 로 등록 (고정, 학습 X)
    """

    def __init__(
        self,
        channels: int,
        share: str = 'channel',
        preset: Optional[str] = 'gelu',
        learn_neg: bool = True,
        learn_pos: bool = True,
        learn_shift: bool = True,
        learn_beta : bool = True,
    ):
        super().__init__()
        assert share in ['channel', 'layer']
        assert preset in ['relu', 'gelu', None]
        self.share = share
        self.preset = preset

        shape = (channels,) if share == 'channel' else ()

        # -----------------------
        # 1) neg / pos (lambda)
        # -----------------------
        init_neg = torch.zeros(shape)
        init_pos = torch.zeros(shape)

        # 초기에는 base activation 그대로 되게끔 모두 0으로.
        # (ReLU/GELU 모양에서 시작해서 점점 보정해 가는 구조)

        if learn_neg:
            self.neg_gsL = nn.Parameter(init_neg)   # 학습용 → 이름에 _gs
        else:
            self.register_buffer("neg", init_neg)  # 고정값

        if learn_pos:
            self.pos_gsL = nn.Parameter(init_pos)
        else:
            self.register_buffer("pos", init_pos)

        # -----------------------
        # 2) shift (center)
        # -----------------------
        init_shift = torch.zeros(shape)

        if learn_shift:
            self.shift_gsL = nn.Parameter(init_shift)
        else:
            self.register_buffer("shift", init_shift)

        # -----------------------
        # 3) beta (gate sharpness)
        # -----------------------
        # 너무 크면 처음부터 ReLU처럼 딱 잘려버리니 1.0 정도로 시작

        init_beta = torch.ones(shape)
        if learn_beta : 
            self.beta_gsL = nn.Parameter(init_beta)
        else : 
            self.register_buffer("beta", init_beta)

        

    # -------------------------------
    # helper: 파라미터를 x shape에 맞게 view
    # -------------------------------
    def _view_params(self, x: torch.Tensor, tg: torch.Tensor):
        """
        CNN: x=(N,C,H,W) -> (1,C,1,1)
        ViT: x=(B,N,D)   -> (1,1,D)
        그 외: 마지막 차원이 feature라고 가정
        """
        if self.share == 'layer':
            view = [1] * x.dim()
            return tg.view(view)

        # 'channel' 공유
        feat = tg.shape[0]
        if x.dim() == 4:
            view = (1, feat, 1, 1)
        elif x.dim() == 3:
            view = (1, 1, feat)
        else:
            view = [1] * (x.dim() - 1) + [feat]
        return tg.view(view)

    def _get_param(self, name_learn: str, name_buf: str) -> torch.Tensor:
        # learn=True 면 *_gs, 아니면 buffer 이름 사용
        if hasattr(self, name_learn):
            return getattr(self, name_learn)
        return getattr(self, name_buf)

    # -------------------------------
    # forward
    # -------------------------------
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # 1) 파라미터 가져오기 & broadcast
    #     neg_raw   = self._get_param("neg_gsL", "neg")
    #     pos_raw   = self._get_param("pos_gsL", "pos")
    #     shift_raw = self._get_param("shift_gsL", "shift")
    #     beta_raw  = self._get_param("beta_gsL", "beta")

    #     neg   = self._view_params(x, neg_raw)
    #     pos   = self._view_params(x, pos_raw)
    #     shift = self._view_params(x, shift_raw)
    #     beta  = self._view_params(x, beta_raw)

    #     # 2) 중심 정렬
    #     xc = x - shift

    #     # 3) base activation (preset에 따라)
    #     if self.preset == 'relu':
    #         base = F.relu(xc)
    #     elif self.preset == 'gelu':
    #         # tanh 근사 GELU
    #         base = 0.5 * xc * (1.0 + torch.tanh(
    #             math.sqrt(2.0 / math.pi) * (xc + 0.044715 * xc**3)
    #         ))
    #     else:
    #         base = xc  # identity

    #     # 4) gate & lambda(x)
    #     gate = torch.sigmoid(beta * xc)
    #     lam = neg + (pos - neg) * gate

    #     # 5) 최종 출력
    #     return base + lam * xc


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape 문제가 나면 안전하게 base(x)만 리턴하기 위한 helper
        def base_only(z: torch.Tensor) -> torch.Tensor:
            if self.preset == 'relu':
                return F.relu(z)
            elif self.preset == 'gelu':
                # tanh 근사 GELU
                return 0.5 * z * (1.0 + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (z + 0.044715 * z**3)
                ))
            else:
                return z

        try:
            # 1) 파라미터 가져오기 & broadcast
            neg_raw   = self._get_param("neg_gsL", "neg")
            pos_raw   = self._get_param("pos_gsL", "pos")
            shift_raw = self._get_param("shift_gsL", "shift")
            beta_raw  = self._get_param("beta_gsL", "beta")

            neg   = self._view_params(x, neg_raw)
            pos   = self._view_params(x, pos_raw)
            shift = self._view_params(x, shift_raw)
            beta  = self._view_params(x, beta_raw)

            # 여기서 shape/broadcast 에러가 날 수 있음
            xc = x - shift

            # base activation (preset에 따라)
            if self.preset == 'relu':
                base = F.relu(xc)
            elif self.preset == 'gelu':
                base = 0.5 * xc * (1.0 + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (xc + 0.044715 * xc**3)
                ))
            else:
                base = xc  # identity

            gate = torch.sigmoid(beta * xc)
            lam = neg + (pos - neg) * gate

            return base + lam * xc

        except RuntimeError as e:
            # 주로 shape / broadcast 문제일 때 여기로 옴
            # 요청대로, 그냥 base(x)만 통과시킴
            # (shift, gate, lam 전부 무시)
            # 디버깅 원하면 여기서 print(e) 찍어도 됨.
            # print("[GS2] shape mismatch, fallback to base only:", e)
            return base_only(x)




# models/gs_model.py 파일의 _infer_channels_from_siblings 함수를
# 아래의 새 버전으로 교체하세요.

def _infer_channels_from_siblings(parent_module: nn.Module, child_name: str):
    """
    Activation 레이어 (GELU, ReLU)의 앞뒤 모듈을 보고
    적용되어야 할 채널(feature) 수를 추론합니다.
    (Linear, Conv2d, BatchNorm 모두 지원)
    """
    
    # 1. 부모가 nn.Sequential인 경우 (이름이 숫자로 됨)
    if isinstance(parent_module, nn.Sequential):
        try:
            idx = int(child_name)
        except ValueError:
            idx = None
            
        if idx is not None:
            # (idx-1) -> 0 방향으로 순회 (Activation 이전 모듈 탐색)
            for j in range(idx - 1, -1, -1):
                sib = parent_module[j]
                if isinstance(sib, nn.Linear):
                    return sib.out_features
                if isinstance(sib, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    return sib.num_features
                if isinstance(sib, nn.Conv2d):
                    return sib.out_channels # 이전 Conv의 out_channels

            # (idx+1) -> 끝 방향으로 순회 (Activation 다음 모듈 탐색)
            for j in range(idx + 1, len(parent_module)):
                sib = parent_module[j]
                if isinstance(sib, nn.Linear):
                    return sib.in_features
                if isinstance(sib, nn.Conv2d):
                    return sib.in_channels # 다음 Conv의 in_channels
            return None

    # 2. 부모가 일반 nn.Module인 경우 (이름이 문자로 됨)
    items = list(parent_module.named_children())
    idx = next((i for i, (n, _) in enumerate(items) if n == child_name), None)
    if idx is None:
        return None

    # (idx-1) -> 0 방향으로 순회 (Activation 이전 모듈 탐색)
    for j in range(idx - 1, -1, -1):
        sib = items[j][1] # (name, module) 튜플에서 모듈 추출
        if isinstance(sib, nn.Linear):
            return sib.out_features
        if isinstance(sib, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return sib.num_features
        if isinstance(sib, nn.Conv2d):
            return sib.out_channels

    # (idx+1) -> 끝 방향으로 순회 (Activation 다음 모듈 탐색)
    for j in range(idx + 1, len(items)):
        sib = items[j][1]
        if isinstance(sib, nn.Linear):
            return sib.in_features
        if isinstance(sib, nn.Conv2d):
            return sib.in_channels
            
    return None




# # =========================
# # 2) 교체 유틸 (GELU/RELU -> GS2)
# # =========================
# def _infer_channels_from_siblings(parent_module: nn.Module, child_name: str):
#     # (원 코드 동일) Linear 주변에서 feature dim 추론
#     if isinstance(parent_module, nn.Sequential):
#         try:
#             idx = int(child_name)
#         except ValueError:
#             idx = None
#         if idx is not None:
#             for j in range(idx - 1, -1, -1):
#                 sib = parent_module[j]
#                 if isinstance(sib, nn.Linear):
#                     return sib.out_features
#             for j in range(idx + 1, len(parent_module)):
#                 sib = parent_module[j]
#                 if isinstance(sib, nn.Linear):
#                     return sib.in_features
#         return None

#     items = list(parent_module.named_children())
#     idx = next((i for i, (n, _) in enumerate(items) if n == child_name), None)
#     if idx is None:
#         return None
#     for j in range(idx - 1, -1, -1):
#         sib = items[j][1]
#         if isinstance(sib, nn.Linear):
#             return sib.out_features
#     for j in range(idx + 1, len(items)):
#         sib = items[j][1]
#         if isinstance(sib, nn.Linear):
#             return sib.in_features
#     return None


def _should_replace_activation(parent_module: nn.Module, parent_prefix: str, child_name: str, adap_list: Optional[List[str]]):
    if not adap_list:
        return True
    full_child = f"{parent_prefix}.{child_name}" if parent_prefix else child_name
    return any(token in full_child for token in adap_list)


def replace_gelu_with_gs2(module: nn.Module, adap_list: Optional[List[str]] = None,
                          module_prefix: str = "", share: str = 'channel', preset: str = 'gelu', ln=False, lt = False, ls = False, lb = False):
    """
    nn.GELU -> GS2(preset='gelu' 기본)
    ViT MLP 등에서 마지막 차원 D 기준 per-channel 적용
    """
    for name, child in module.named_children():
        full = f"{module_prefix}.{name}" if module_prefix else name
        if isinstance(child, nn.GELU):
            if not _should_replace_activation(module, module_prefix, name, adap_list):
                continue

            channels = _infer_channels_from_siblings(module, name)
            if channels is None:
                channels = getattr(module, 'embed_dim', None) or getattr(module, 'hidden_dim', None) or 768
                print(f"[GS2] WARN: infer channels failed at {full}; default {channels}")

            new_act = GS2(channels=channels, share=share, preset=preset, learn_neg = ln, learn_pos = lt, learn_shift = ls, learn_beta = lb)
            if isinstance(module, nn.Sequential):
                module[int(name)] = new_act
            else:
                setattr(module, name, new_act)
            print(f"[GS2] Replaced GELU at '{full}' → GS2({channels}, preset={preset})")
        else:
            replace_gelu_with_gs2(child, adap_list=adap_list, module_prefix=full, share=share, preset=preset,  ln=ln, lt=lt, ls=ls, lb = lb)


def replace_relu_with_gs2(module: nn.Module, adap_list: Optional[List[str]] = None,
                          module_prefix: str = "", share: str = 'channel', preset: str = 'relu', ln=False, lt = False, ls = False, lb = False):
    """
    nn.ReLU -> GS2(preset='relu' 기본)
    CNN에서 채널 축(C) 기준 per-channel 적용
    """
    # print(module, 'module!!')
    for name, child in module.named_children():
        
        full = f"{module_prefix}.{name}" if module_prefix else name

        if isinstance(child, nn.ReLU):
            if adap_list is not None and not any(full.startswith(prefix) for prefix in adap_list):
                continue

            channels = _infer_channels_from_siblings(module, name)
            if channels is None:
                # CNN 모듈에서 parent에 out_channels가 직접 없을 수 있으므로 보수적 디폴트
                channels = getattr(module, 'out_channels', None) or 64
                print(f"[GS2] WARN: infer channels failed at {full}; default {channels}")

            new_act = GS2(channels=channels, share=share, preset=preset , learn_neg = ln, learn_pos = lt, learn_shift = ls, learn_beta = lb)
            if isinstance(module, nn.Sequential):
                module[int(name)] = new_act
            else:
                setattr(module, name, new_act)
            print(f"[GS2] Replaced ReLU at '{full}' → GS2({channels}, preset={preset})")
        else:
            replace_relu_with_gs2(child, adap_list=adap_list, module_prefix=full, share=share, preset=preset,  ln=ln, lt=lt, ls=ls, lb= lb)


# =========================
# 3) 모델별 래퍼
# =========================
def vit_b_16_gs2(model=None, adap_layer: int = 12, share: str = 'channel' , ln=False, lt = False, ls = False, lb= False):
    """
    ViT의 blocks.{0..adap_layer-1} 범위 내 GELU를 GS2로 교체 (preset='gelu')
    """
    # targets = [f'blocks.{i}.' for i in range(adap_layer)]
    targets = [f'model.blocks.{i}.' for i in range(adap_layer )]
    # targets = [f'blocks.{adap_layer}.']
    replace_gelu_with_gs2(model, adap_list=targets, module_prefix="", share=share, preset='gelu' ,  ln=ln, lt=lt, ls=ls, lb =lb)
    return model


def cnn_gs2(model=None, adap_layer: int = 4, share: str = 'channel' , ln=False, lt = False, ls = False, lb = False):
    """
    ResNet 등 CNN의 ReLU를 GS2로 교체 (preset='relu')
    adap_layer: layer1~layer{adap_layer} 범위
    """
    # targets = [f'layer{i}.' for i in range(1, adap_layer + 1)]
    targets = [f'block{i}.' for i in range(1, adap_layer+1)] ## WRN28
    print(targets,'targets!!!!')
    replace_relu_with_gs2(model, adap_list=targets, module_prefix="", share=share, preset='relu',  ln=ln, lt=lt, ls=ls, lb = lb)
    return model




def resnet50_gs2(model=None, adap_layer: int = 4, share: str = 'channel' , ln=False, lt = False, ls = False, lb = False):
    """
    ResNet 등 CNN의 ReLU를 GS2로 교체 (preset='relu')
    adap_layer: layer1~layer{adap_layer} 범위
    """
    # targets = [f'layer{i}.' for i in range(1, adap_layer + 1)]
    targets = [f'model.layer{i}.' for i in range(0, adap_layer+1)] ## WRN28
    targets.append('model.relu')
    print(targets,' targets!!')
    replace_relu_with_gs2(model, adap_list=targets, module_prefix="", share=share, preset='relu',  ln=ln, lt=lt, ls=ls, lb = lb)
    return model



def resnext_gs2(model=None, adap_layer: int = 4, share: str = 'channel' , ln=False, lt = False, ls = False, lb = False):
    
    
    targets = [f'stage_{i}.' for i in range(1, adap_layer+1)] ## WRN28

    replace_relu_with_gs2(model, adap_list=targets, module_prefix="", share=share, preset='relu',  ln=ln, lt=lt, ls=ls, lb = lb)
    return model



def build_generic_stage_prefixes(model: nn.Module, max_stage: int = 4,
                                 keys=('layer', 'stage'),
                                 include_stage_root: bool = True):
    """
    모델에서 'layer1', 'layer2', ... 또는 'stage_1', 'stage_2', ... 같은
    stage 모듈을 자동으로 탐색해서 프리픽스를 만듭니다.
    - max_stage: layer/stage 인덱스의 상한 (1..max_stage)
    - include_stage_root: 'layer1.' / 'stage_1.' 자체도 포함할지
    반환 예: ['layer1.', 'layer1.0.', 'layer1.1.', 'layer2.', 'layer2.0.', ...]
    """
    prefixes = []
    top = dict(model.named_children())  # 최상위 자식만

    # 후보 stage 이름들 스캔
    for k in keys:
        for i in range(1, max_stage + 1):
            # layer1 / stage_1 두 패턴 모두 시도
            cand_names = [f"{k}{i}", f"{k}_{i}"]
            for cand in cand_names:
                if cand not in top:
                    continue
                stage = top[cand]
                if include_stage_root:
                    prefixes.append(f"{cand}.")

                # 보통은 Sequential이 많음
                if isinstance(stage, nn.Sequential):
                    for j, _ in enumerate(stage):
                        prefixes.append(f"{cand}.{j}.")
                else:
                    # Sequential 이 아니라면, 자식 중 숫자 이름만 추가
                    for name, _ in stage.named_children():
                        if re.fullmatch(r"\d+", name):
                            prefixes.append(f"{cand}.{name}.")
                        # 숫자가 아니어도 stage 루트 프리픽스는 이미 포함
    return prefixes



# =========================
# 4) 예시 구동
# =========================
if __name__ == "__main__":
    # ResNet: layer1만 교체
    m = resnet50()
    m = cnn_gs2(m, adap_layer=1, share='channel')

    # ViT-B/16: 앞 2블록만 교체
    # vit = vit_b_16(weights=None)
    # vit = vit_b_16_gs2(vit, adap_layer=2, share='channel')
