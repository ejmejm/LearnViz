import pygame
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import numpy.typing as npt
from .network_structure import NetworkSpec, LayerSpec
from .serialization import StepData


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
        
        # View transformation state
        self.zoom = 1.0
        self.is_panning = False
        self.last_mouse_pos = None
        
        # Calculate weight ranges and initialize UI
        self.weight_range = self._calculate_weight_range()
        self._initialize_ui()


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
        # UI elements
        self.slider_rect = pygame.Rect(
            200,
            self.layout.height - 60,
            self.layout.width - 250,
            20,
        )
        self.play_button_rect = pygame.Rect(
            50,
            self.layout.height - 60,
            40,
            40,
        )
        self.pause_button_rect = pygame.Rect(
            100,
            self.layout.height - 60,
            40,
            40,
        )
        
        # Colors
        self.colors = {
            'background': (0, 0, 0),
            'neuron': (240, 240, 240),
            'neutral': (50, 50, 50),
            'positive': (0, 0, 255),
            'negative': (255, 0, 0),
            'ui': (200, 200, 200),
        }


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
        if self.slider_rect.collidepoint(mouse_pos):
            rel_x = (mouse_pos[0] - self.slider_rect.left) / self.slider_rect.width
            self.current_step = int(rel_x * self.num_steps)
            self.current_step = max(0, min(self.current_step, self.num_steps - 1))
        elif self.play_button_rect.collidepoint(mouse_pos):
            self.is_playing = True
        elif self.pause_button_rect.collidepoint(mouse_pos):
            self.is_playing = False


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


    def _world_to_screen(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        x = int(pos[0] * self.zoom + self.pan_offset[0])
        y = int(pos[1] * self.zoom + self.pan_offset[1])
        return (x, y)


    def _draw_network(self, step_data: StepData):
        """Draw the network with current weights and biases"""
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
        
        # Draw neurons with antialiasing
        for layer in self.neuron_positions:
            for pos in layer:
                screen_pos = self._world_to_screen(pos)
                # Draw filled circle
                pygame.draw.circle(
                    self.screen,
                    self.colors['neuron'],
                    screen_pos,
                    int(self.layout.neuron_radius * self.zoom),
                    0
                )


    def _draw_controls(self):
        """Draw the UI controls"""
        # Draw slider
        pygame.draw.rect(self.screen, self.colors['ui'], self.slider_rect, 2)
        
        # Draw slider handle
        handle_pos = (
            self.slider_rect.left + 
            (self.current_step / self.num_steps) * self.slider_rect.width
        )
        pygame.draw.rect(
            self.screen,
            self.colors['ui'],
            (handle_pos - 5, self.slider_rect.top - 5, 10, 30)
        )
        
        # Draw play/pause buttons
        pygame.draw.rect(self.screen, self.colors['ui'], self.play_button_rect, 2)
        pygame.draw.rect(self.screen, self.colors['ui'], self.pause_button_rect, 2)
        
        # Draw play/pause icons
        if not self.is_playing:
            # Play triangle
            pygame.draw.polygon(
                self.screen,
                self.colors['ui'],
                [
                    (self.play_button_rect.left + 10, self.play_button_rect.top + 10),
                    (self.play_button_rect.left + 10, self.play_button_rect.bottom - 10),
                    (self.play_button_rect.right - 10, self.play_button_rect.centery)
                ]
            )
        
        # Pause bars
        pygame.draw.rect(
            self.screen,
            self.colors['ui'],
            (self.pause_button_rect.left + 10, self.pause_button_rect.top + 10, 5, 20)
        )
        pygame.draw.rect(
            self.screen,
            self.colors['ui'],
            (self.pause_button_rect.right - 15, self.pause_button_rect.top + 10, 5, 20)
        )


    def _update_control_positions(self):
        """Update UI control positions based on current screen size"""
        screen_width, screen_height = self.screen.get_size()
        
        self.slider_rect = pygame.Rect(
            200,
            screen_height - 60,
            screen_width - 250,
            20,
        )
        self.play_button_rect = pygame.Rect(
            50,
            screen_height - 60,
            40,
            40,
        )
        self.pause_button_rect = pygame.Rect(
            100,
            screen_height - 60,
            40,
            40,
        )


    def run(self):
        """Main visualization loop"""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            current_time = pygame.time.get_ticks()
            
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
                
                elif event.type == pygame.MOUSEMOTION:
                    # Handle panning
                    if self.is_panning and self.last_mouse_pos is not None:
                        delta_x = event.pos[0] - self.last_mouse_pos[0]
                        delta_y = event.pos[1] - self.last_mouse_pos[1]
                        self.pan_offset[0] += delta_x
                        self.pan_offset[1] += delta_y
                        self.last_mouse_pos = event.pos
                    
                    # Handle slider dragging
                    elif event.buttons[0] and self.slider_rect.collidepoint(event.pos):
                        rel_x = (event.pos[0] - self.slider_rect.left) / self.slider_rect.width
                        self.current_step = int(rel_x * self.num_steps)
                        self.current_step = max(0, min(self.current_step, self.num_steps - 1))
            
            # Update step if playing
            if self.is_playing and current_time - self.last_update > self.update_interval:
                self.current_step = (self.current_step + 1) % self.num_steps
                self.last_update = current_time
            
            # Draw
            self.screen.fill(self.colors['background'])
            self._draw_network(self.get_step_data(self.current_step))
            self._draw_controls()
            
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
                has_bias=True
            ),
            LayerSpec(
                layer_type='linear',
                input_size=4,
                output_size=2,
                has_bias=True
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
            ]
        )
    
    # Create and run visualizer
    visualizer = NetworkVisualizer(
        network_spec=test_network,
        num_steps=num_steps,
        get_step_data=get_random_step_data
    )
    visualizer.run()
