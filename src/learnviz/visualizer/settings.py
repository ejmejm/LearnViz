from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class LayoutConfig:
    """Configuration for the visualization layout"""
    width: int = 1200
    height: int = 800
    top_margin: int = 100  # For loss display
    bottom_margin: int = 100  # For controls
    horizontal_margin: int = 50
    neuron_radius: int = 30
    min_layer_spacing: int = 60
    max_layer_spacing: int = 300
    min_neuron_spacing: int = 70
    weight_line_width: int = 5
    zoom_speed: float = 0.1
    button_width: int = 150
    button_height: int = 30
    button_margin: int = 10
    button_text_size: int = 16


DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    'background': (0, 0, 0),
    'neuron': (220, 220, 220),
    'neutral': (50, 50, 50),
    'positive': (0, 0, 255),
    'negative': (255, 0, 0),
    'ui': (200, 200, 200),
    'slider_progress': (255, 0, 0),
    'button_active': (240, 240, 240),
    'button_inactive': (100, 100, 100),
    'button_text_active': (50, 50, 50),
    'button_text_inactive': (240, 240, 240),
    'activation_positive': (0, 0, 255),
    'activation_negative': (255, 0, 0),
    'loss_min': (50, 200, 50),    # Green for minimum loss
    'loss_max': (255, 165, 0),    # Orange for maximum loss
    'loss_box': (50, 50, 50),     # Gray background for loss box
}


class UIConfig:
    """Configuration for UI behavior"""
    KEY_REPEAT_DELAY: int = 300  # ms before key starts repeating
    KEY_REPEAT_INTERVAL: int = 50  # ms between repeats
    UPDATE_INTERVAL: int = 50  # ms between steps when playing
    FPS: int = 60