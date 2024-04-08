"""
A version of ConvNeXt modified from: https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html
Made for use with 3D input data using depth 1 convolutional layers.
Models can load pretrained ImageNet weights from torchvision.
Includes a few added forward functions to enable using the models for UNets, as feature encoding layers for transformers,
or to extract features for use alongside tabular data.
Note: Intermediate activations consume a lot of memory so you may want to run the models within torch.no_grad()
"""

from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.ops.misc import Conv3dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models.convnext import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights


__all__ = [
    "ConvNeXt3D",
    "convnext3D_tiny",
    "convnext3D_small",
    "convnext3D_base",
    "convnext3D_large",
]


class LayerNorm3d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 7, 7), padding=(0, 3, 3), groups=dim, bias=True),
            Permute([0, 2, 3, 4, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 4, 1, 2, 3]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm3d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv3dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=(1, 4, 4),
                stride=(1, 4, 4),
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv3d(cnf.input_channels, cnf.out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward_features(self, x: Tensor, depth: int) -> Tensor:
        """
        Forwards the convolutional features from the block specified by depth (max 7)
        blocks alternate between processing features (odds) and downscaling features (evens)
        """
        x = self.features[:depth+1](x)
        return x

    def forward_penultimate(self, x: Tensor) -> Tensor:
        """
        Forwards the input to the final Linear layer.
        May be useful for combining with tabular features.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier[:-1](x)
        return x

    def forward_encoder(self, x: Tensor) -> Tuple[Tensor]:
        """
        Forwards the activations before each downsampling step
        """
        x = self.features[0](x)  # 1/4 input resolution
        x0 = self.features[1](x)  # 1/4
        x = self.features[2](x0)  # 1/8
        x1 = self.features[3](x)  # 1/8
        x = self.features[4](x1)  # 1/16
        x2 = self.features[5](x)  # 1/16
        x = self.features[6](x2)  # 1/32
        x3 = self.features[7](x)  # 1/32
        return x0, x1, x2, x3

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _convnext(
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ConvNeXt:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress, check_hash=True)
        for param_tensor in state_dict:
            layer_index = param_tensor.split('.')
            match len(layer_index):  # lens 3, 4, 6
                case 3:
                    layer = getattr(getattr(model, layer_index[0]), layer_index[1])
                case 4:
                    layer = getattr(getattr(getattr(model, layer_index[0]), layer_index[1]), layer_index[2])
                case 6:
                    layer = getattr(getattr(getattr(getattr(getattr(model, layer_index[0]), layer_index[1]), layer_index[2]), layer_index[3]), layer_index[4])
                case _:
                    layer = None  # default case should never happen

            if isinstance(layer, torch.nn.Conv3d):
                if 'weight' in param_tensor:
                    state_dict[param_tensor] = torch.unsqueeze(state_dict[param_tensor], dim=2)

            if 'layer_scale' in layer_index:
                state_dict[param_tensor] = torch.unsqueeze(state_dict[param_tensor], dim=1)

        model.load_state_dict(state_dict)
    return model


@register_model()
@handle_legacy_interface(weights=("pretrained", ConvNeXt_Tiny_Weights.IMAGENET1K_V1))
def convnext3D_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """
    weights = ConvNeXt_Tiny_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ConvNeXt_Small_Weights.IMAGENET1K_V1))
def convnext3D_small(*, weights: Optional[ConvNeXt_Small_Weights] = None, progress: bool = True, **kwargs: Any ) -> ConvNeXt:
    """ConvNeXt Small model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Small_Weights
        :members:
    """
    weights = ConvNeXt_Small_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ConvNeXt_Base_Weights.IMAGENET1K_V1))
def convnext3D_base(*, weights: Optional[ConvNeXt_Base_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Base model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Base_Weights
        :members:
    """
    weights = ConvNeXt_Base_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ConvNeXt_Large_Weights.IMAGENET1K_V1))
def convnext3D_large(*, weights: Optional[ConvNeXt_Large_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Large model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Large_Weights
        :members:
    """
    weights = ConvNeXt_Large_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)
