import pygame
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict
import numpy.typing as npt
from .network_structure import NetworkSpec, LayerSpec
from .serialization import StepData
from math import sqrt


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


@dataclass
class Button:
    """Represents a toggle button in the UI"""
    rect: pygame.Rect
    text: str
    is_active: bool = True


class NetworkVisualizer:
    def __init__(
        self,
        network_spec: NetworkSpec,
        num_steps: int,
        get_step_data: Callable[[int], StepData],
        layout: Optional[LayoutConfig] = None,
    ):
        pygame.init()
        self.layout = layout or LayoutConfig()
        self.network_spec = network_spec
        self.num_steps = num_steps
        self.get_step_data = get_step_data
        
        # Initialize display with RESIZABLE flag
        self.screen = pygame.display.set_mode(
            size = (self.layout.width, self.layout.height),
            flags = pygame.RESIZABLE,
        )
        pygame.display.set_caption('Neural Network Visualizer')
        
        # Calculate neuron positions and initialize view
        self.neuron_positions = self._calculate_neuron_positions()
        self._initialize_view()
        
        # UI state
        self.current_step = 0
        self.is_playing = False
        self.last_update = 0
        self.update_interval = 50  # ms between steps when playing
        self.is_dragging_slider = False  # New state for slider dragging
        self.key_repeat_delay = 300  # ms before key starts repeating
        self.key_repeat_interval = 50  # ms between repeats
        self.last_key_time = 0
        self.key_held_since = 0
        
        # View transformation state
        self.zoom = 1.0
        self.is_panning = False
        self.last_mouse_pos = None
        
        # Calculate weight ranges and initialize UI
        self.weight_range = self._calculate_weight_range()
        self._initialize_ui()
        
        # Initialize buttons
        self.buttons: Dict[str, Button] = {}
        self._initialize_buttons()
        
        # Button states
        self.show_weight_changes = False
        
        # Add these new attributes
        self.hovered_weight = None
        self.font = pygame.font.Font(None, 24)
        self.hovered_neuron = None


    def _initialize_view(self):
        """Initialize camera position to center on network"""
        network_center_x = (
            self.neuron_positions[0][0][0] + 
            self.neuron_positions[-1][0][0]
        ) / 2
        network_center_y = sum(
            pos[1] for layer in self.neuron_positions 
            for pos in layer
        ) / sum(len(layer) for layer in self.neuron_positions)
        
        self.pan_offset = [
            self.layout.width / 2 - network_center_x,
            self.layout.height / 2 - network_center_y,
        ]


    def _initialize_ui(self):
        """Initialize UI elements and colors"""
        # Calculate total control width (play button + slider)
        play_button_width = 40
        slider_width = self.layout.width - 400  # Reduced width to leave margins
        total_width = play_button_width + 10 + slider_width  # 10px spacing between button and slider
        
        # Calculate left position to center everything
        start_x = (self.layout.width - total_width) // 2
        play_button_y = self.layout.height - 50
        
        # Position play button
        self.play_pause_button_rect = pygame.Rect(
            start_x,
            play_button_y,
            play_button_width,
            40,
        )
        
        # Position slider next to play button
        slider_y = play_button_y + (self.play_pause_button_rect.height // 2) - 2
        self.slider_rect = pygame.Rect(
            start_x + play_button_width + 10,  # 10px spacing after play button
            slider_y,
            slider_width,
            4,
        )
        
        # Colors
        self.colors = {
            'background': (0, 0, 0),
            'neuron': (220, 220, 220),
            'neutral': (50, 50, 50),
            'positive': (0, 0, 255),
            'negative': (255, 0, 0),
            'ui': (200, 200, 200),
            'slider_progress': (255, 0, 0),  # Red color for progress
            'button_active': (240, 240, 240),
            'button_inactive': (100, 100, 100),
            'button_text_active': (50, 50, 50),
            'button_text_inactive': (240, 240, 240),
            'activation_positive': (0, 0, 255),  # Green for positive activations
            'activation_negative': (255, 0, 0),  # Orange for negative activations
        }


    def _initialize_buttons(self):
        """Initialize toggle buttons in top-left corner based on available data"""
        # Get first step data to determine what's available
        first_step = self.get_step_data(0)
        
        # Start with weights button which is always present
        button_names = ['Toggle Weights']
        
        # Add buttons based on available data
        if first_step.inputs is not None:
            button_names.append('Toggle Inputs')
        if first_step.activations is not None:
            button_names.append('Toggle Activations')
        if first_step.loss is not None:
            button_names.append('Toggle Loss')
            
        # Add buttons for per-weight values
        if first_step.per_weight_values:
            for key in first_step.per_weight_values:
                button_name = f"Toggle {key.replace('_', ' ').title()}"
                button_names.append(button_name)
                
        # Add buttons for extra values
        if first_step.extra_values:
            for key in first_step.extra_values:
                button_name = f"Toggle {key.replace('_', ' ').title()}"
                button_names.append(button_name)
        
        # Create buttons
        for i, name in enumerate(button_names):
            self.buttons[name] = Button(
                rect=pygame.Rect(
                    self.layout.button_margin,
                    self.layout.button_margin + i * (self.layout.button_height + self.layout.button_margin),
                    self.layout.button_width,
                    self.layout.button_height
                ),
                text=name
            )


    def _handle_zoom(self, mouse_pos: Tuple[int, int], zoom_in: bool):
        """Handle zoom in/out at mouse position"""
        old_zoom = self.zoom
        if zoom_in:
            self.zoom *= (1 + self.layout.zoom_speed)
        else:
            self.zoom /= (1 + self.layout.zoom_speed)
        
        # Calculate world position of mouse
        mouse_world_x = (mouse_pos[0] - self.pan_offset[0]) / old_zoom
        mouse_world_y = (mouse_pos[1] - self.pan_offset[1]) / old_zoom
        
        # Update pan offset to maintain mouse position
        self.pan_offset[0] = mouse_pos[0] - mouse_world_x * self.zoom
        self.pan_offset[1] = mouse_pos[1] - mouse_world_y * self.zoom


    def _handle_ui_click(self, mouse_pos: Tuple[int, int]):
        """Handle clicks on UI elements"""
        # Check button clicks
        for button in self.buttons.values():
            if button.rect.collidepoint(mouse_pos):
                button.is_active = not button.is_active
                return
                
        # Check if click is within vertical range of slider handle
        handle_radius = 8
        if (abs(mouse_pos[1] - self.slider_rect.centery) <= handle_radius and 
            self.slider_rect.left <= mouse_pos[0] <= self.slider_rect.right):
            self.is_dragging_slider = True
            rel_x = (mouse_pos[0] - self.slider_rect.left) / self.slider_rect.width
            self.current_step = int(rel_x * self.num_steps)
            self.current_step = max(0, min(self.current_step, self.num_steps - 1))
        elif self.play_pause_button_rect.collidepoint(mouse_pos):
            self.is_playing = not self.is_playing


    def _calculate_neuron_positions(self) -> List[List[Tuple[int, int]]]:
        """Calculate the positions of all neurons in the network"""
        layers = [self.network_spec.input_size] + [
            layer.output_size for layer in self.network_spec.layers
        ]
        
        available_height = (
            self.layout.height - self.layout.top_margin - self.layout.bottom_margin
        )
        
        # Calculate layer spacing based on available width
        available_width = self.layout.width - 2 * self.layout.horizontal_margin
        num_gaps = len(layers) - 1
        layer_spacing = min(
            self.layout.max_layer_spacing,
            max(
                self.layout.min_layer_spacing,
                available_width / num_gaps
            )
        )
        
        positions = []
        for layer_idx, layer_size in enumerate(layers):
            layer_positions = []
            layer_height = (layer_size - 1) * self.layout.min_neuron_spacing
            start_y = (
                self.layout.height - self.layout.bottom_margin - 
                (available_height - layer_height) // 2
            )
            
            x = (
                self.layout.horizontal_margin + 
                layer_idx * layer_spacing
            )
            
            for neuron_idx in range(layer_size):
                y = start_y - neuron_idx * self.layout.min_neuron_spacing
                layer_positions.append((int(x), int(y)))
            
            positions.append(layer_positions)
            
        return positions


    def _calculate_weight_range(self) -> Tuple[float, float]:
        """Calculate the min and max weights across all steps"""
        min_weight = float('inf')
        max_weight = float('-inf')
        
        for step in range(self.num_steps):
            data = self.get_step_data(step)
            for weights in data.weights:
                min_weight = min(min_weight, weights.min())
                max_weight = max(max_weight, weights.max())
            for bias in data.biases:
                if bias is not None:
                    min_weight = min(min_weight, bias.min())
                    max_weight = max(max_weight, bias.max())
                    
        return min_weight, max_weight


    def _get_weight_color(self, weight: float) -> Tuple[int, int, int]:
        """Get the color for a weight based on its value"""
        if not self.buttons['Toggle Weights'].is_active:
            return self.colors['neutral']
            
        max_abs = max(abs(self.weight_range[0]), abs(self.weight_range[1]))
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


    def _get_activation_color(self, value: float) -> Tuple[int, int, int]:
        """Get color for an activation/input value"""
        if value is None:
            return self.colors['neuron']
        
        # Scale intensity based on absolute value
        intensity = min(abs(value), 1.0)
        
        if value > 0:
            # Interpolate between neutral and positive
            return tuple(
                int(n * (1 - intensity) + p * intensity)
                for n, p in zip(self.colors['neuron'], self.colors['activation_positive'])
            )
        else:
            # Interpolate between neutral and negative
            return tuple(
                int(n * (1 - intensity) + p * intensity)
                for n, p in zip(self.colors['neuron'], self.colors['activation_negative'])
            )


    def _world_to_screen(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        x = int(pos[0] * self.zoom + self.pan_offset[0])
        y = int(pos[1] * self.zoom + self.pan_offset[1])
        return (x, y)


    def _screen_to_world(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        x = (pos[0] - self.pan_offset[0]) / self.zoom
        y = (pos[1] - self.pan_offset[1]) / self.zoom
        return (x, y)


    def _find_hovered_weight(self, mouse_pos: Tuple[int, int], step_data: StepData):
        """Find the weight that the mouse is hovering over"""
        if not self.buttons['Toggle Weights'].is_active:
            self.hovered_weight = None
            return

        world_pos = self._screen_to_world(mouse_pos)
        
        # Check each weight connection
        for layer_idx, (weights, biases) in enumerate(zip(step_data.weights, step_data.biases)):
            # Check weights between layers
            for i, pos1 in enumerate(self.neuron_positions[layer_idx]):
                for j, pos2 in enumerate(self.neuron_positions[layer_idx + 1]):
                    # Calculate distance from point to line segment
                    p1 = pygame.Vector2(pos1)
                    p2 = pygame.Vector2(pos2)
                    p3 = pygame.Vector2(world_pos)
                    
                    # Vector from p1 to p2
                    line_vec = p2 - p1
                    line_length = line_vec.length()
                    if line_length == 0:
                        continue
                    
                    # Vector from p1 to mouse
                    point_vec = p3 - p1
                    
                    # Calculate projection
                    t = max(0, min(1, point_vec.dot(line_vec) / (line_length * line_length)))
                    
                    # Find closest point on line
                    closest = p1 + line_vec * t
                    
                    # Check if mouse is close enough to line
                    distance = (p3 - closest).length()
                    if distance < 5 / self.zoom:  # Adjust threshold as needed
                        self.hovered_weight = {
                            'value': weights[j, i],
                            'pos': (closest.x, closest.y)
                        }
                        return

            # Check bias connections if present
            if biases is not None:
                bias_x = (
                    self.neuron_positions[layer_idx][0][0] + 
                    self.layout.neuron_radius * 2
                )
                bias_y = (
                    self.layout.height - self.layout.bottom_margin + 
                    self.layout.neuron_radius
                )
                bias_pos = (bias_x, bias_y)
                
                for j, pos in enumerate(self.neuron_positions[layer_idx + 1]):
                    p1 = pygame.Vector2(bias_pos)
                    p2 = pygame.Vector2(pos)
                    p3 = pygame.Vector2(world_pos)
                    
                    line_vec = p2 - p1
                    line_length = line_vec.length()
                    if line_length == 0:
                        continue
                    
                    point_vec = p3 - p1
                    t = max(0, min(1, point_vec.dot(line_vec) / (line_length * line_length)))
                    closest = p1 + line_vec * t
                    
                    distance = (p3 - closest).length()
                    if distance < 5 / self.zoom:
                        self.hovered_weight = {
                            'value': biases[j],
                            'pos': (closest.x, closest.y)
                        }
                        return
        
        self.hovered_weight = None


    def _find_hovered_neuron(self, mouse_pos: Tuple[int, int], step_data: StepData):
        """Find the neuron that the mouse is hovering over and its value"""
        # Reset hover states at start
        self.hovered_neuron = None
        self.hovered_weight = None  # Reset weight hover when checking neurons
        
        world_pos = self._screen_to_world(mouse_pos)
        
        # Check each neuron
        for layer_idx, layer_positions in enumerate(self.neuron_positions):
            for neuron_idx, pos in enumerate(layer_positions):
                # Calculate distance from mouse to neuron center
                distance = sqrt(
                    (world_pos[0] - pos[0])**2 + 
                    (world_pos[1] - pos[1])**2
                )
                
                if distance < self.layout.neuron_radius:
                    value = None
                    if layer_idx == 0 and step_data.inputs is not None:
                        value = step_data.inputs[neuron_idx]
                    elif layer_idx > 0 and step_data.activations is not None:
                        value = step_data.activations[layer_idx - 1][neuron_idx]
                    
                    if value is not None:
                        self.hovered_neuron = {
                            'value': value,
                            'pos': pos
                        }
                    return
        
        # Check bias neurons if no regular neuron is hovered
        for layer_idx, biases in enumerate(step_data.biases):
            if biases is not None:
                bias_x = (
                    self.neuron_positions[layer_idx][0][0] + 
                    self.layout.neuron_radius * 2
                )
                bias_y = (
                    self.layout.height - self.layout.bottom_margin + 
                    self.layout.neuron_radius
                )
                bias_pos = (bias_x, bias_y)
                
                distance = sqrt(
                    (world_pos[0] - bias_pos[0])**2 + 
                    (world_pos[1] - bias_pos[1])**2
                )
                
                if distance < self.layout.neuron_radius:
                    self.hovered_neuron = {
                        'value': 1.0,  # Bias neurons always have value 1
                        'pos': bias_pos
                    }
                    return
        
        # Only find hovered weight if no neuron is hovered
        self._find_hovered_weight(mouse_pos, step_data)


    def _draw_network(self, step_data: StepData):
        """Draw the network with current weights, biases, and activations"""
        # Draw weights
        for layer_idx, (weights, biases) in enumerate(zip(step_data.weights, step_data.biases)):
            # Draw weights between layers
            for i, pos1 in enumerate(self.neuron_positions[layer_idx]):
                for j, pos2 in enumerate(self.neuron_positions[layer_idx + 1]):
                    weight = weights[j, i]
                    color = self._get_weight_color(weight)
                    # Use antialiasing for lines
                    pygame.draw.aaline(
                        self.screen,
                        color,
                        self._world_to_screen(pos1),
                        self._world_to_screen(pos2)
                    )
                    # Draw thicker non-antialiased line if needed
                    if self.layout.weight_line_width > 1:
                        pygame.draw.line(
                            self.screen,
                            color,
                            self._world_to_screen(pos1),
                            self._world_to_screen(pos2),
                            max(1, int(self.layout.weight_line_width * self.zoom))
                        )
            
            # Draw bias connections if present
            if biases is not None:
                bias_x = (
                    self.neuron_positions[layer_idx][0][0] + 
                    self.layout.neuron_radius * 2
                )
                bias_y = (
                    self.layout.height - self.layout.bottom_margin + 
                    self.layout.neuron_radius
                )
                bias_pos = (bias_x, bias_y)
                
                for j, pos in enumerate(self.neuron_positions[layer_idx + 1]):
                    color = self._get_weight_color(biases[j])
                    # Use antialiasing for lines
                    pygame.draw.aaline(
                        self.screen,
                        color,
                        self._world_to_screen(bias_pos),
                        self._world_to_screen(pos)
                    )
                    # Draw thicker non-antialiased line if needed
                    if self.layout.weight_line_width > 1:
                        pygame.draw.line(
                            self.screen,
                            color,
                            self._world_to_screen(bias_pos),
                            self._world_to_screen(pos),
                            max(1, int(self.layout.weight_line_width * self.zoom))
                        )
                
                # Draw bias neuron with antialiasing
                screen_pos = self._world_to_screen(bias_pos)
                pygame.draw.circle(
                    self.screen,
                    self.colors['neuron'],
                    screen_pos,
                    int(self.layout.neuron_radius * self.zoom),
                    0
                )
        
        # Draw neurons with activations
        for layer_idx, layer in enumerate(self.neuron_positions):
            for neuron_idx, pos in enumerate(layer):
                screen_pos = self._world_to_screen(pos)
                color = self.colors['neuron']
                
                # Color based on input/activation if available and buttons are active
                if layer_idx == 0 and step_data.inputs is not None:
                    if self.buttons['Toggle Inputs'].is_active:
                        color = self._get_activation_color(step_data.inputs[neuron_idx])
                elif layer_idx > 0 and step_data.activations is not None:
                    if self.buttons['Toggle Activations'].is_active:
                        color = self._get_activation_color(
                            step_data.activations[layer_idx - 1][neuron_idx]
                        )
                
                # Draw filled circle
                pygame.draw.circle(
                    self.screen,
                    color,
                    screen_pos,
                    int(self.layout.neuron_radius * self.zoom),
                    0
                )


    def _draw_controls(self):
        """Draw the UI controls"""
        # Calculate progress position
        progress_width = (self.current_step / self.num_steps) * self.slider_rect.width
        
        # Draw progress part of slider (red)
        progress_rect = pygame.Rect(
            self.slider_rect.left,
            self.slider_rect.top,
            progress_width,
            self.slider_rect.height
        )
        pygame.draw.rect(
            self.screen,
            self.colors['slider_progress'],
            progress_rect,
            0
        )
        
        # Draw remaining part of slider (white)
        remaining_rect = pygame.Rect(
            self.slider_rect.left + progress_width,
            self.slider_rect.top,
            self.slider_rect.width - progress_width,
            self.slider_rect.height
        )
        pygame.draw.rect(
            self.screen,
            self.colors['ui'],
            remaining_rect,
            0
        )
        
        # Draw slider handle (red circle)
        handle_pos = (
            self.slider_rect.left + progress_width,
            self.slider_rect.centery
        )
        pygame.draw.circle(
            self.screen,
            self.colors['slider_progress'],
            handle_pos,
            8
        )
        
        # Draw play/pause icon based on state
        if not self.is_playing:
            # Play triangle
            pygame.draw.polygon(
                self.screen,
                self.colors['ui'],
                [
                    (self.play_pause_button_rect.left + 10, self.play_pause_button_rect.top + 10),
                    (self.play_pause_button_rect.left + 10, self.play_pause_button_rect.bottom - 10),
                    (self.play_pause_button_rect.right - 10, self.play_pause_button_rect.centery)
                ]
            )
        else:
            # Pause bars
            pygame.draw.rect(
                self.screen,
                self.colors['ui'],
                (self.play_pause_button_rect.left + 10, self.play_pause_button_rect.top + 10, 5, 20)
            )
            pygame.draw.rect(
                self.screen,
                self.colors['ui'],
                (self.play_pause_button_rect.right - 15, self.play_pause_button_rect.top + 10, 5, 20)
            )


    def _draw_buttons(self):
        """Draw all toggle buttons"""
        font = pygame.font.Font(None, self.layout.button_text_size)
        
        for button in self.buttons.values():
            # Draw button background
            color = self.colors['button_active'] if button.is_active else self.colors['button_inactive']
            pygame.draw.rect(self.screen, color, button.rect)
            
            # Draw button text
            text_color = self.colors['button_text_active'] if button.is_active else self.colors['button_text_inactive']
            text_surface = font.render(button.text, True, text_color)
            text_rect = text_surface.get_rect(center=button.rect.center)
            self.screen.blit(text_surface, text_rect)


    def _update_control_positions(self):
        """Update UI control positions based on current screen size"""
        screen_width, screen_height = self.screen.get_size()
        
        # Calculate total control width (play button + slider)
        play_button_width = 40
        slider_width = screen_width - 400  # Reduced width to leave margins
        total_width = play_button_width + 10 + slider_width  # 10px spacing between button and slider
        
        # Calculate left position to center everything
        start_x = (screen_width - total_width) // 2
        play_button_y = screen_height - 50
        
        # Position play button
        self.play_pause_button_rect = pygame.Rect(
            start_x,
            play_button_y,
            play_button_width,
            40,
        )
        
        # Position slider next to play button
        slider_y = play_button_y + (self.play_pause_button_rect.height // 2) - 2
        self.slider_rect = pygame.Rect(
            start_x + play_button_width + 10,  # 10px spacing after play button
            slider_y,
            slider_width,
            4,
        )


    def _step_forward(self):
        """Move one step forward"""
        self.current_step = min(self.current_step + 1, self.num_steps - 1)


    def _step_backward(self):
        """Move one step backward"""
        self.current_step = max(self.current_step - 1, 0)


    def run(self):
        """Main visualization loop"""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Get current mouse position and find hovered elements
            mouse_pos = pygame.mouse.get_pos()
            step_data = self.get_step_data(self.current_step)
            self._find_hovered_neuron(mouse_pos, step_data)  # This now handles both neurons and weights
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(
                        size = (event.w, event.h),
                        flags = pygame.RESIZABLE,
                    )
                    self._update_control_positions()
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.is_playing = not self.is_playing
                    elif event.key in (pygame.K_RIGHT, pygame.K_d, pygame.K_LEFT, pygame.K_a):
                        self.key_held_since = current_time
                        if event.key in (pygame.K_RIGHT, pygame.K_d):
                            self._step_forward()
                        else:
                            self._step_backward()
                
                elif event.type == pygame.KEYUP:
                    if event.key in (pygame.K_RIGHT, pygame.K_d, pygame.K_LEFT, pygame.K_a):
                        self.key_held_since = 0
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    
                    if event.button == 3:  # Right click
                        self.is_panning = True
                        self.last_mouse_pos = mouse_pos
                    elif event.button == 4:  # Mouse wheel up
                        self._handle_zoom(mouse_pos, zoom_in=True)
                    elif event.button == 5:  # Mouse wheel down
                        self._handle_zoom(mouse_pos, zoom_in=False)
                    elif event.button == 1:  # Left click
                        self._handle_ui_click(mouse_pos)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:  # Right click release
                        self.is_panning = False
                    elif event.button == 1:  # Left click release
                        self.is_dragging_slider = False
                
                elif event.type == pygame.MOUSEMOTION:
                    # Handle panning
                    if self.is_panning and self.last_mouse_pos is not None:
                        delta_x = event.pos[0] - self.last_mouse_pos[0]
                        delta_y = event.pos[1] - self.last_mouse_pos[1]
                        self.pan_offset[0] += delta_x
                        self.pan_offset[1] += delta_y
                        self.last_mouse_pos = event.pos
                    
                    # Handle slider dragging - now works even when mouse is off slider
                    elif self.is_dragging_slider:
                        rel_x = (event.pos[0] - self.slider_rect.left) / self.slider_rect.width
                        rel_x = max(0, min(1, rel_x))  # Clamp between 0 and 1
                        self.current_step = int(rel_x * self.num_steps)
                        self.current_step = max(0, min(self.current_step, self.num_steps - 1))
            
            # Handle key repeat for arrow/WASD keys
            if self.key_held_since > 0:
                time_held = current_time - self.key_held_since
                if time_held > self.key_repeat_delay and current_time - self.last_key_time > self.key_repeat_interval:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                        self._step_forward()
                    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                        self._step_backward()
                    self.last_key_time = current_time
            
            # Update step if playing
            if self.is_playing and current_time - self.last_update > self.update_interval:
                if self.current_step < self.num_steps - 1:
                    self.current_step += 1
                else:
                    self.is_playing = False  # Stop playing when we reach the end
                self.last_update = current_time
            
            # Draw
            self.screen.fill(self.colors['background'])
            self._draw_network(step_data)
            self._draw_controls()
            self._draw_buttons()
            
            # Draw weight value if hovering
            if self.hovered_weight:
                screen_pos = self._world_to_screen(self.hovered_weight['pos'])
                text = f"{self.hovered_weight['value']:.3f}"
                text_surface = self.font.render(text, True, self.colors['ui'])
                text_rect = text_surface.get_rect()
                
                # Offset the text position above and to the right of the cursor
                offset_x = 5
                offset_y = -4
                text_rect.bottomleft = (screen_pos[0] + offset_x, screen_pos[1] + offset_y)
                
                # Add background for better visibility
                padding = 4
                bg_rect = text_rect.inflate(padding * 2, padding * 2)
                pygame.draw.rect(self.screen, self.colors['background'], bg_rect)
                self.screen.blit(text_surface, text_rect)
            
            if self.hovered_neuron:
                screen_pos = self._world_to_screen(self.hovered_neuron['pos'])
                text = f"{self.hovered_neuron['value']:.3f}"
                text_surface = self.font.render(text, True, self.colors['ui'])
                text_rect = text_surface.get_rect(center=screen_pos)
                
                # Add background for better visibility
                padding = 4
                bg_rect = text_rect.inflate(padding * 2, padding * 2)
                pygame.draw.rect(self.screen, self.colors['background'], bg_rect)
                self.screen.blit(text_surface, text_rect)
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()


if __name__ == '__main__':
    # Create a simple test network specification
    test_network = NetworkSpec(
        layers=[
            LayerSpec(
                layer_type='linear',
                input_size=3,
                output_size=4,
                has_bias=True,
            ),
            LayerSpec(
                layer_type='linear',
                input_size=4,
                output_size=2,
                has_bias=True,
            )
        ],
        input_size=3,
        output_size=2
    )
    
    # Create random training data
    num_steps = 100
    
    def get_random_step_data(step: int) -> StepData:
        """Generate random step data for testing"""
        rng = np.random.default_rng(step)
        return StepData(
            weights=[
                rng.normal(0, 1, (4, 3)),  # First layer weights
                rng.normal(0, 1, (2, 4))   # Second layer weights
            ],
            biases=[
                rng.normal(0, 1, (4,)),    # First layer biases
                rng.normal(0, 1, (2,))     # Second layer biases
            ],
            loss=rng.normal(0, 1),
            inputs=rng.normal(0, 1, (3,)),
            activations=[
                rng.normal(0, 1, (4,)),
                rng.normal(0, 1, (2,))
            ],
            per_weight_values={
                'step_size': [
                    rng.normal(0, 1, (4, 3)),
                    rng.normal(0, 1, (2, 4))
                ],
            },
            extra_values={
                'custom_metric': rng.uniform(0.5, 1.0)
            }
        )
    
    # Create and run visualizer
    visualizer = NetworkVisualizer(
        network_spec=test_network,
        num_steps=num_steps,
        get_step_data=get_random_step_data
    )
    visualizer.run()
