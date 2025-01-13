import pygame
from typing import Dict, Tuple, List, Optional
import numpy as np
from ..serialization import StepData
from .settings import LayoutConfig, DEFAULT_COLORS


class GraphicsManager:
    """Handles drawing and coordinate transformations"""
    def __init__(self):
        self.zoom = 1.0
        self.pan_offset = [0, 0]
        self.colors = DEFAULT_COLORS.copy()  # Make a copy to avoid modifying the original
        self.font = pygame.font.Font(None, 24)
        self.hovered_weight = None
        self.hovered_neuron = None
        self.is_panning = False
        self.last_mouse_pos = None
        self.loss_range = (float('inf'), float('-inf'))  # (min_loss, max_loss)

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

    def _calculate_loss_range(self, step_data: StepData) -> None:
        """Update min/max loss values"""
        if step_data.loss is not None:
            self.loss_range = (
                min(self.loss_range[0], step_data.loss),
                max(self.loss_range[1], step_data.loss)
            )

    def draw_loss_box(
        self, 
        screen: pygame.Surface, 
        step_data: StepData,
        neuron_positions: List[List[Tuple[int, int]]],
        layout_config: LayoutConfig
    ) -> None:
        """Draw the loss box to the right of the network"""
        if step_data.loss is None:
            return

        # Update loss range
        self._calculate_loss_range(step_data)
        
        # Calculate box position (right of network, vertically centered)
        rightmost_neuron_x = max(pos[0] for layer in neuron_positions for pos in layer)
        network_center_y = sum(
            pos[1] for layer in neuron_positions for pos in layer
        ) / sum(len(layer) for layer in neuron_positions)
        
        base_font_size = 24  # Base font size before zoom
        scaled_font_size = max(1, int(base_font_size * self.zoom))
        scaled_font = pygame.font.Font(None, scaled_font_size)
        
        box_width = 150
        box_height = 40
        margin = 80
        
        box_x = rightmost_neuron_x + margin
        box_y = network_center_y - box_height / 2
        
        # Convert to screen coordinates
        screen_pos = self.world_to_screen((box_x, box_y))
        box_rect = pygame.Rect(
            screen_pos[0],
            screen_pos[1],
            int(box_width * self.zoom),
            int(box_height * self.zoom)
        )
        
        # Draw box background with darker color
        darker_box_color = tuple(max(0, c - 20) for c in self.colors['loss_box'])
        pygame.draw.rect(screen, darker_box_color, box_rect)
        
        # Calculate loss color based on min/max range
        if self.loss_range[0] != self.loss_range[1]:
            progress = (step_data.loss - self.loss_range[0]) / (self.loss_range[1] - self.loss_range[0])
            color = tuple(
                int(min_val + (max_val - min_val) * progress)
                for min_val, max_val in zip(
                    self.colors['loss_min'],
                    self.colors['loss_max']
                )
            )
        else:
            color = self.colors['loss_min']
        
        # Draw loss text with scaled font
        text = f"Loss: {step_data.loss:.3g}"
        text_surface = scaled_font.render(text, True, color)
        text_rect = text_surface.get_rect(center=box_rect.center)
        screen.blit(text_surface, text_rect) 

    def draw_value_boxes(
        self, 
        screen: pygame.Surface, 
        step_data: StepData,
        neuron_positions: List[List[Tuple[int, int]]],
        layout_config: LayoutConfig,
        ui_manager
    ) -> None:
        """Draw the loss and extra value boxes to the right of the network"""
        # Collect all values to display
        values_to_display = []
        
        # Always add loss first if it exists and is toggled
        if step_data.loss is not None and ui_manager.get_button_state('Toggle Loss'):
            self._calculate_loss_range(step_data)
            values_to_display.append(('Loss', step_data.loss, True))
            
        # Add extra values if they exist and are toggled
        if step_data.extra_values:
            for key, value in step_data.extra_values.items():
                button_name = f"Toggle {key.replace('_', ' ').title()}"
                if ui_manager.get_button_state(button_name):
                    values_to_display.append((key.replace('_', ' ').title(), value, False))
        
        if not values_to_display:
            return
            
        # Prepare font and calculate scaled dimensions
        base_font_size = 24
        scaled_font_size = max(1, int(base_font_size * self.zoom))
        scaled_font = pygame.font.Font(None, scaled_font_size)
        
        # Calculate box dimensions based on longest label
        box_height = int(40 * self.zoom)  # Scale box height with zoom
        box_spacing = int(10 * self.zoom)  # Scale spacing with zoom
        margin = int(80 * self.zoom)  # Scale margin with zoom
        value_margin = int(10 * self.zoom)  # Scale value margin with zoom
        
        # Calculate box width based on longest label at current zoom level
        longest_label = max((label for label, _, _ in values_to_display), key=len)
        label_width = scaled_font.size(longest_label + ': ')[0]
        box_width = label_width + int(20 * self.zoom)  # Add scaled padding
        
        # Calculate total height of all boxes
        total_height = len(values_to_display) * box_height + (len(values_to_display) - 1) * box_spacing
        
        # Calculate starting positions
        rightmost_neuron_x = max(pos[0] for layer in neuron_positions for pos in layer)
        network_center_y = sum(
            pos[1] for layer in neuron_positions for pos in layer
        ) / sum(len(layer) for layer in neuron_positions)
        
        start_y = network_center_y - total_height / 2
        box_x = rightmost_neuron_x + margin
        
        # Draw each value box and its corresponding value
        for i, (label, value, is_loss) in enumerate(values_to_display):
            box_y = start_y + i * (box_height + box_spacing)
            
            # Convert to screen coordinates
            screen_pos = self.world_to_screen((box_x, box_y))
            box_rect = pygame.Rect(
                screen_pos[0],
                screen_pos[1],
                box_width,  # Already scaled width
                box_height  # Already scaled height
            )
            
            # Draw box background
            darker_box_color = tuple(max(0, c - 20) for c in self.colors['loss_box'])
            pygame.draw.rect(screen, darker_box_color, box_rect)
            
            # Draw label (always white, right-aligned)
            label_text = f"{label}: "
            label_surface = scaled_font.render(label_text, True, self.colors['ui'])
            label_rect = label_surface.get_rect()
            label_rect.right = box_rect.right - int(5 * self.zoom)  # Scale padding
            label_rect.centery = box_rect.centery
            screen.blit(label_surface, label_rect)
            
            # Draw value (outside box)
            value_text = f"{value:.3g}"
            
            # Determine value color
            if is_loss and self.loss_range[0] != self.loss_range[1]:
                progress = (value - self.loss_range[0]) / (self.loss_range[1] - self.loss_range[0])
                color = tuple(
                    int(min_val + (max_val - min_val) * progress)
                    for min_val, max_val in zip(
                        self.colors['loss_min'],
                        self.colors['loss_max']
                    )
                )
            else:
                color = self.colors['ui']
            
            value_surface = scaled_font.render(value_text, True, color)
            value_rect = value_surface.get_rect()
            value_rect.left = box_rect.right + value_margin
            value_rect.centery = box_rect.centery
            screen.blit(value_surface, value_rect) 