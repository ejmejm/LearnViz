from dataclasses import dataclass
from enum import Enum
from typing import Optional


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