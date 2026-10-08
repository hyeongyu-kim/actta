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


class GS2(nn.Module):
    """
    GateSpline-2 (shift + gated-slope residual on top of base activation)
    
    [추가 기능]
    - 초기 파라미터값 (neg, pos, shift)을 'init_*_gs' 버퍼에 저장합니다.
    - init_param_consistency() 메서드는 현재 학습된 파라미터와 초기값 간의 L1 거리를 반환합니다.
    """

    def __init__(
        self,
        channels: int,
        share: str = 'channel',
        preset: Optional[str] = 'gelu',
        learn_neg: bool = True,
        learn_pos: bool = True,
        learn_shift: bool = True,
        consis_weight: float = 0.01
    ):
        super().__init__()
        assert share in ['channel', 'layer']
        assert preset in ['relu', 'gelu', None]
        self.share = share
        self.preset = preset
        self.learn_neg = learn_neg
        self.learn_pos = learn_pos
        self.learn_shift = learn_shift
        self.consis_weight = consis_weight
        shape = (channels,) if share == 'channel' else ()

        # --- 1) neg / pos (lambda) ---
        init_neg = torch.zeros(shape, device="cuda")
        init_pos = torch.zeros(shape, device="cuda")
        self.register_buffer("init_neg_gs", init_neg.clone())
        self.register_buffer("init_pos_gs", init_pos.clone())

        if learn_neg:
            self.neg_gsL = nn.Parameter(init_neg)
        else:
            self.register_buffer("neg", init_neg)

        if learn_pos:
            self.pos_gsL = nn.Parameter(init_pos)
        else:
            self.register_buffer("pos", init_pos)

        # --- 2) shift (center) ---
        init_shift = torch.zeros(shape, device="cuda")
        self.register_buffer("init_shift_gs", init_shift.clone())

        if learn_shift:
            self.shift_gsL = nn.Parameter(init_shift)
        else:
            self.register_buffer("shift", init_shift)

        # --- 3) beta (gate sharpness) ---
        init_beta = torch.ones(shape, device="cuda")
        self.register_buffer("beta", init_beta)
        

    def init_param_consistency(self) -> torch.Tensor:
        l1_loss = torch.tensor(0.0, device=self._get_param("neg_gsL", "neg").device)
        
        # 1. neg 일관성
        if self.learn_neg:
            current_neg = self.neg_gsL
            init_neg = self.init_neg_gs
            l1_loss = l1_loss + torch.sum(torch.abs(current_neg - init_neg))
        
        # 2. pos 일관성
        if self.learn_pos:
            current_pos = self.pos_gsL
            init_pos = self.init_pos_gs
            l1_loss = l1_loss + torch.sum(torch.abs(current_pos - init_pos))
        
        # 3. shift 일관성
        if self.learn_shift:
            current_shift = self.shift_gsL
            init_shift = self.init_shift_gs
            l1_loss = l1_loss + torch.sum(torch.abs(current_shift - init_shift))
            
        return l1_loss * self.consis_weight


    # -------------------------------
    # helper: 파라미터를 x shape에 맞게 view (생략되지 않음)
    # -------------------------------
    def _view_params(self, x: torch.Tensor, tg: torch.Tensor):
        if self.share == 'layer':
            view = [1] * x.dim()
            return tg.view(view)

        feat = tg.shape[0]
        if x.dim() == 4:
            view = (1, feat, 1, 1)
        elif x.dim() == 3:
            view = (1, 1, feat)
        else:
            view = [1] * (x.dim() - 1) + [feat]
        return tg.view(view)

    def _get_param(self, name_learn: str, name_buf: str) -> torch.Tensor:
        if hasattr(self, name_learn):
            return getattr(self, name_learn)
        return getattr(self, name_buf)

    # -------------------------------
    # forward (생략되지 않음)
    # -------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 파라미터 가져오기 & broadcast
        neg_raw   = self._get_param("neg_gsL", "neg")
        pos_raw   = self._get_param("pos_gsL", "pos")
        shift_raw = self._get_param("shift_gsL", "shift")
        beta_raw  = self._get_param("beta_gsL", "beta")

        neg   = self._view_params(x, neg_raw)
        pos   = self._view_params(x, pos_raw)
        shift = self._view_params(x, shift_raw)
        beta  = self._view_params(x, beta_raw)

        # 2) 중심 정렬
        xc = x - shift

        # 3) base activation (preset에 따라)
        if self.preset == 'relu':
            base = F.relu(xc)
        elif self.preset == 'gelu':
            # tanh 근사 GELU
            base = 0.5 * xc * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (xc + 0.044715 * xc**3)
            ))
        else:
            base = xc  # identity

        # 4) gate & lambda(x)
        gate = torch.sigmoid(beta * xc)
        lam = neg + (pos - neg) * gate

        # 5) 최종 출력
        return base + lam * xc







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



def _should_replace_activation(parent_module: nn.Module, parent_prefix: str, child_name: str, adap_list: Optional[List[str]]):
    if not adap_list:
        return True
    full_child = f"{parent_prefix}.{child_name}" if parent_prefix else child_name
    return any(token in full_child for token in adap_list)


def replace_gelu_with_gs2(module: nn.Module, adap_list: Optional[List[str]] = None,
                          module_prefix: str = "", share: str = 'channel', preset: str = 'gelu', ln=False, lt = False, ls = False, consis_weight = 0.01):
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

            new_act = GS2(channels=channels, share=share, preset=preset, learn_neg = ln, learn_pos = lt, learn_shift = ls, consis_weight = consis_weight)
            if isinstance(module, nn.Sequential):
                module[int(name)] = new_act
            else:
                setattr(module, name, new_act)
            print(f"[GS2] Replaced GELU at '{full}' → GS2({channels}, preset={preset})")
        else:
            replace_gelu_with_gs2(child, adap_list=adap_list, module_prefix=full, share=share, preset=preset,  ln=ln, lt=lt, ls=ls, consis_weight = consis_weight)


def replace_relu_with_gs2(module: nn.Module, adap_list: Optional[List[str]] = None,
                          module_prefix: str = "", share: str = 'channel', preset: str = 'relu', ln=False, lt = False, ls = False, consis_weight = 0.01):
    """
    nn.ReLU -> GS2(preset='relu' 기본)
    CNN에서 채널 축(C) 기준 per-channel 적용
    """
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

            new_act = GS2(channels=channels, share=share, preset=preset , learn_neg = ln, learn_pos = lt, learn_shift = ls, consis_weight = consis_weight)
            if isinstance(module, nn.Sequential):
                module[int(name)] = new_act
            else:
                setattr(module, name, new_act)
            print(f"[GS2] Replaced ReLU at '{full}' → GS2({channels}, preset={preset})")
        else:
            replace_relu_with_gs2(child, adap_list=adap_list, module_prefix=full, share=share, preset=preset,  ln=ln, lt=lt, ls=ls)


# =========================
# 3) 모델별 래퍼
# =========================
def vit_b_16_gs2(model=None, adap_layer: int = 12, share: str = 'channel' , ln=False, lt = False, ls = False, consis_weight = 0.01):
    """
    ViT의 blocks.{0..adap_layer-1} 범위 내 GELU를 GS2로 교체 (preset='gelu')
    """
    print(consis_weight)
    # targets = [f'blocks.{i}.' for i in range(adap_layer)]
    targets = [f'model.blocks.{i}.' for i in range(adap_layer )]
    # targets = [f'blocks.{adap_layer}.']
    replace_gelu_with_gs2(model, adap_list=targets, module_prefix="", share=share, preset='gelu' ,  ln=ln, lt=lt, ls=ls, consis_weight = consis_weight)
    return model


def cnn_gs2(model=None, adap_layer: int = 4, share: str = 'channel' , ln=False, lt = False, ls = False, consis_weight = 0.01):
    """
    ResNet 등 CNN의 ReLU를 GS2로 교체 (preset='relu')
    adap_layer: layer1~layer{adap_layer} 범위
    """
    # targets = [f'layer{i}.' for i in range(1, adap_layer + 1)]
    targets = [f'block{i}.' for i in range(1, adap_layer+1)] ## WRN28
    replace_relu_with_gs2(model, adap_list=targets, module_prefix="", share=share, preset='relu',  ln=ln, lt=lt, ls=ls, consis_weight = consis_weight)
    return model



def _replace_gelu_relu_by_count_recursive(
    module: nn.Module,
    max_replacements: int,
    replacement_counter: List[int],  # 만난 활성화 개수 (교체 + 스킵 포함)
    replaced_count: List[int],       # 실제 교체된 개수만 카운트
    start_point: int = 0,            # N번째 activation 이후부터 교체 시작
    module_prefix: str = "",
    share: str = 'channel',
    ln: bool = False,
    lt: bool = False,
    ls: bool = False,
    preset: str = None,
    consis_weight: float = 0.01,
):
    """
    Activation(ReLU/GELU)을 앞에서부터 순회하면서
    start_point 이후의 것부터 max_replacements 만큼 GS2로 교체하는 함수
    """
    if replaced_count[0] >= max_replacements:
        return

    for name, child in module.named_children():
        if replaced_count[0] >= max_replacements:
            break

        full_name = f"{module_prefix}.{name}" if module_prefix else name

        # ======================================================
        # 1) Activation 체크
        # ======================================================
        if isinstance(child, nn.ReLU) or isinstance(child, nn.GELU):

            # 현재 몇 번째 activation인지 기록
            replacement_counter[0] += 1

            # 아직 교체할 타이밍이 아님
            if replacement_counter[0] < start_point:
                continue

            # ======================================================
            # 2) 채널 추론
            # ======================================================
            channels = _infer_channels_from_siblings(module, name)
            if channels is None:
                # fallback
                channels = getattr(module, "out_channels", None) \
                           or getattr(module, "embed_dim", None) \
                           or getattr(module, "hidden_dim", None) \
                           or 64
                print(f"[GS2] WARN: channel inference failed at {full_name}; default={channels}")

            # ======================================================
            # 3) preset 자동 설정
            # ======================================================
            if preset is None:
                act_preset = 'relu' if isinstance(child, nn.ReLU) else 'gelu'
            else:
                act_preset = preset

            # ======================================================
            # 4) GS2 생성
            # ======================================================
            new_activation = GS2(
                channels=channels,
                share=share,
                preset=act_preset,
                learn_neg=ln,
                learn_pos=lt,
                learn_shift=ls,
                consis_weight=consis_weight,
            )

            # ======================================================
            # 5) 모듈에 삽입
            # ======================================================
            if isinstance(module, nn.Sequential):
                module[int(name)] = new_activation
            else:
                setattr(module, name, new_activation)

            replaced_count[0] += 1

            print(
                f"({replaced_count[0]}/{max_replacements}) "
                f"Replaced '{full_name}' (#{replacement_counter[0]} activation) "
                f"with GS2(ch={channels}, preset={act_preset})"
            )

        else:
            # ======================================================
            # 자식 모듈 DFS
            # ======================================================
            _replace_gelu_relu_by_count_recursive(
                child,
                max_replacements=max_replacements,
                replacement_counter=replacement_counter,
                replaced_count=replaced_count,
                start_point=start_point,
                module_prefix=full_name,
                share=share,
                ln=ln,
                lt=lt,
                ls=ls,
                preset=preset,
                consis_weight=consis_weight,
            )


def model_gs2_by_count(
    model: nn.Module,
    num_replacements_total: int,   # 교체할 총 개수
    start_point: int = 1,          # N번째 활성화부터 교체 시작 (1-based)
    share: str = 'channel',
    ln: bool = False,
    lt: bool = False,
    ls: bool = False,
    preset: str = None,            # 'relu', 'gelu', or None(auto)
    consis_weight: float = 0.01,
):
    """
    모델 전체를 앞에서부터 순회하여,
    start_point번째 activation부터 num_replacements_total개 만큼
    ReLU/GELU를 GS2로 교체한다.
    """
    if num_replacements_total <= 0:
        print("No activations will be replaced (num_replacements_total <= 0).")
        return model

    replacement_counter = [0]  # 지금까지 만난 활성화 개수
    replaced_count = [0]       # 실제 교체된 개수

    _replace_gelu_relu_by_count_recursive(
        model,
        max_replacements=num_replacements_total,
        replacement_counter=replacement_counter,
        replaced_count=replaced_count,
        start_point=start_point,
        module_prefix="",
        share=share,
        ln=ln,
        lt=lt,
        ls=ls,
        preset=preset,
        consis_weight=consis_weight,
    )

    print(
        f"\n=== GS2 Replacement Complete ===\n"
        f"Replaced {replaced_count[0]} activations (requested {num_replacements_total})\n"
        f"Start point: {start_point}\n"
        f"================================\n"
    )

    return model


from torchvision.models.resnet import Bottleneck


class custom_res_block(nn.Module):
    expansion = 4

    def __init__(self, init_block: Bottleneck):
        """
        init_block: torchvision.models.resnet.Bottleneck 객체
        """
        super().__init__()
        inplace = (init_block.relu.inplace if hasattr(init_block, 'relu') else True)

        # =============================
        #   1) Conv / BN 그대로 상속
        # =============================
        self.conv1 = init_block.conv1
        self.bn1   = init_block.bn1
        self.relu1 = nn.ReLU(inplace=inplace)
        self.conv2 = init_block.conv2
        self.bn2   = init_block.bn2
        self.relu2 = nn.ReLU(inplace=inplace)
        self.conv3 = init_block.conv3
        self.bn3   = init_block.bn3
        self.relu3 = nn.ReLU(inplace=inplace)
        self.downsample = init_block.downsample
        self.stride = init_block.stride

    def forward(self, x):
        identity = x

        # conv1 → bn1 → relu1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # conv2 → bn2 → relu2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # conv3 → bn3
        out = self.conv3(out)
        out = self.bn3(out)

        # skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        # 마지막 relu3
        out = self.relu3(out)

        return out

def replace_bottlenecks(module: nn.Module):
    """
    재귀적으로 모든 submodule을 탐색하여 
    torchvision Bottleneck을 custom_res_block 으로 교체한다.
    """
    for name, child in module.named_children():

        # 1) child가 Bottleneck이면 custom_res_block으로 교체
        if isinstance(child, Bottleneck):
            setattr(module, name, custom_res_block(child))

        # 2) child 내부도 다시 탐색 (재귀)
        else:
            replace_bottlenecks(child)


def convert_resnet50_with_custom_block(model: nn.Module) -> nn.Module:
    """
    torchvision resnet50의 모든 Bottleneck 블록을 custom_res_block으로 교체.
    """
    replace_bottlenecks(model)
    return model


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
