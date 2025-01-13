from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class LayerType(Enum):
    LINEAR = 1


@dataclass
class LayerSpec:
    """Specification for a neural network layer"""
    layer_type: LayerType
    input_size: int
    output_size: int
    has_bias: bool = True
    activation: Optional[str] = None


@dataclass
class NetworkSpec:
    """Specification for a neural network"""
    layers: List[LayerSpec]
    input_size: int
    output_size: int

    def __init__(
        self,
        layers: List[LayerSpec],
        input_size: Optional[int] = None,
        output_size: Optional[int] = None
    ):
        self.layers = layers
        self.input_size = input_size or layers[0].input_size
        self.output_size = output_size or layers[-1].output_size
