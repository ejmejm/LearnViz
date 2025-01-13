import pygame
from typing import Dict, Tuple, List, Optional
import numpy as np
from ..serialization import StepData
from .ui import LayoutConfig


class GraphicsManager:
    """Handles drawing and coordinate transformations"""
    def __init__(self):
        self.zoom = 1.0
        self.pan_offset = [0, 0]
        self.colors = self._create_color_scheme()
        self.font = pygame.font.Font(None, 24)
        self.hovered_weight = None
        self.hovered_neuron = None
        self.is_panning = False
        self.last_mouse_pos = None

    def _create_color_scheme(self) -> Dict[str, Tuple[int, int, int]]:
        """Create the default color scheme"""
        return {
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
        }

    def world_to_screen(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        x = int(pos[0] * self.zoom + self.pan_offset[0])
        y = int(pos[1] * self.zoom + self.pan_offset[1])
        return (x, y)

    def screen_to_world(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        x = (pos[0] - self.pan_offset[0]) / self.zoom
        y = (pos[1] - self.pan_offset[1]) / self.zoom
        return (x, y)

    def get_weight_color(self, weight: float, weight_range: Tuple[float, float]) -> Tuple[int, int, int]:
        """Get the color for a weight based on its value"""
        max_abs = max(abs(weight_range[0]), abs(weight_range[1]))
        intensity = min(abs(weight) / max_abs, 1.0)
        
        if weight > 0:
            # Interpolate between neutral gray and blue
            return tuple(
                int(n * (1 - intensity) + b * intensity)
                for n, b in zip(self.colors['neutral'], self.colors['positive'])
            )
        else:
            # Interpolate between neutral gray and red
            return tuple(
                int(n * (1 - intensity) + r * intensity)
                for n, r in zip(self.colors['neutral'], self.colors['negative'])
            )

    def get_activation_color(self, value: float) -> Tuple[int, int, int]:
        """Get color for an activation/input value"""
        if value is None:
            return self.colors['neuron']
        
        intensity = min(abs(value), 1.0)
        
        if value > 0:
            return tuple(
                int(n * (1 - intensity) + p * intensity)
                for n, p in zip(self.colors['neuron'], self.colors['activation_positive'])
            )
        else:
            return tuple(
                int(n * (1 - intensity) + p * intensity)
                for n, p in zip(self.colors['neuron'], self.colors['activation_negative'])
            )

    def draw_network(
        self, 
        screen: pygame.Surface,
        step_data: StepData,
        neuron_positions: List[List[Tuple[int, int]]],
        layout_config: LayoutConfig,
        show_weights: bool = True,
        show_inputs: bool = True, 
        show_activations: bool = True
    ) -> None:
        """Draw the network with current weights, biases, and activations"""
        # Update the weight drawing logic
        if show_weights:
            weight_range = self._calculate_weight_range(step_data)
        else:
            weight_range = (-1, 1)  # Default range when weights are hidden
        
        # Draw weights
        for layer_idx, (layer_weights, layer_biases) in enumerate(zip(step_data.weights, step_data.biases)):
            # Draw weights between layers
            for i, pos1 in enumerate(neuron_positions[layer_idx]):
                for j, pos2 in enumerate(neuron_positions[layer_idx + 1]):
                    if show_weights:
                        weight = layer_weights[j, i]
                        color = self.get_weight_color(weight, weight_range)
                    else:
                        color = self.colors['neutral']

                    # Draw antialiased line
                    pygame.draw.aaline(
                        screen,
                        color,
                        self.world_to_screen(pos1),
                        self.world_to_screen(pos2)
                    )
                    
                    # Draw thicker line if needed
                    if layout_config.weight_line_width > 1:
                        pygame.draw.line(
                            screen,
                            color,
                            self.world_to_screen(pos1),
                            self.world_to_screen(pos2),
                            max(1, int(layout_config.weight_line_width * self.zoom))
                        )

            # Draw bias connections if present
            if layer_biases is not None:
                bias_x = (
                    neuron_positions[layer_idx][0][0] + 
                    layout_config.neuron_radius * 2
                )
                bias_y = (
                    layout_config.height - layout_config.bottom_margin + 
                    layout_config.neuron_radius
                )
                bias_pos = (bias_x, bias_y)
                
                for j, pos in enumerate(neuron_positions[layer_idx + 1]):
                    color = self.get_weight_color(layer_biases[j], weight_range) if show_weights else self.colors['neutral']
                    pygame.draw.aaline(
                        screen,
                        color,
                        self.world_to_screen(bias_pos),
                        self.world_to_screen(pos)
                    )
                    if layout_config.weight_line_width > 1:
                        pygame.draw.line(
                            screen,
                            color,
                            self.world_to_screen(bias_pos),
                            self.world_to_screen(pos),
                            max(1, int(layout_config.weight_line_width * self.zoom))
                        )
                
                # Draw bias neuron
                screen_pos = self.world_to_screen(bias_pos)
                pygame.draw.circle(
                    screen,
                    self.colors['neuron'],
                    screen_pos,
                    int(layout_config.neuron_radius * self.zoom)
                )

        # Draw neurons
        for layer_idx, layer in enumerate(neuron_positions):
            for neuron_idx, pos in enumerate(layer):
                screen_pos = self.world_to_screen(pos)
                color = self.colors['neuron']
                
                # Color based on input/activation if available and buttons are active
                if layer_idx == 0 and step_data.inputs is not None and show_inputs:
                    color = self.get_activation_color(step_data.inputs[neuron_idx])
                elif layer_idx > 0 and step_data.activations is not None and show_activations:
                    color = self.get_activation_color(
                        step_data.activations[layer_idx - 1][neuron_idx]
                    )
                
                pygame.draw.circle(
                    screen,
                    color,
                    screen_pos,
                    int(layout_config.neuron_radius * self.zoom)
                )

    def draw_hover_info(self, screen: pygame.Surface) -> None:
        """Draw hover information for weights and neurons"""
        if self.hovered_weight:
            screen_pos = self.world_to_screen(self.hovered_weight['pos'])
            text = f"{self.hovered_weight['value']:.3f}"
            text_surface = self.font.render(text, True, self.colors['ui'])
            text_rect = text_surface.get_rect()
            
            offset_x, offset_y = 5, -4
            text_rect.bottomleft = (screen_pos[0] + offset_x, screen_pos[1] + offset_y)
            
            padding = 4
            bg_rect = text_rect.inflate(padding * 2, padding * 2)
            pygame.draw.rect(screen, self.colors['background'], bg_rect)
            screen.blit(text_surface, text_rect)
        
        if self.hovered_neuron:
            screen_pos = self.world_to_screen(self.hovered_neuron['pos'])
            text = f"{self.hovered_neuron['value']:.3f}"
            text_surface = self.font.render(text, True, self.colors['ui'])
            text_rect = text_surface.get_rect(center=screen_pos)
            
            padding = 4
            bg_rect = text_rect.inflate(padding * 2, padding * 2)
            pygame.draw.rect(screen, self.colors['background'], bg_rect)
            screen.blit(text_surface, text_rect) 

    def _handle_zoom(self, mouse_pos: Tuple[int, int], zoom_in: bool) -> None:
        """Handle zoom in/out at mouse position"""
        old_zoom = self.zoom
        if zoom_in:
            self.zoom *= (1 + 0.1)  # Using default zoom_speed
        else:
            self.zoom /= (1 + 0.1)
        
        # Calculate world position of mouse
        mouse_world_x = (mouse_pos[0] - self.pan_offset[0]) / old_zoom
        mouse_world_y = (mouse_pos[1] - self.pan_offset[1]) / old_zoom
        
        # Update pan offset to maintain mouse position
        self.pan_offset[0] = mouse_pos[0] - mouse_world_x * self.zoom
        self.pan_offset[1] = mouse_pos[1] - mouse_world_y * self.zoom

    def _calculate_weight_range(self, step_data: StepData) -> Tuple[float, float]:
        """Calculate the min and max weights"""
        min_weight = float('inf')
        max_weight = float('-inf')
        
        for weights in step_data.weights:
            min_weight = min(min_weight, weights.min())
            max_weight = max(max_weight, weights.max())
        for bias in step_data.biases:
            if bias is not None:
                min_weight = min(min_weight, bias.min())
                max_weight = max(max_weight, bias.max())
                
        return min_weight, max_weight 