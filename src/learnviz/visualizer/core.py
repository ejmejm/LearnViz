import pygame
from typing import Callable, Optional, List, Tuple, Dict
from math import sqrt
import numpy as np
from .ui import LayoutConfig, UIManager
from .graphics import GraphicsManager
from ..network_structure import NetworkSpec, LayerSpec
from ..serialization import StepData


class NetworkVisualizer:
    """Main class for visualizing neural network training"""
    def __init__(
        self,
        network_spec: NetworkSpec,
        num_steps: int,
        get_step_data: Callable[[int], StepData],
        layout: Optional[LayoutConfig] = None,
    ):
        pygame.init()
        self.layout = layout or LayoutConfig()
        self.ui = UIManager(self.layout)
        self.graphics = GraphicsManager()
        
        # Initialize display and state
        self.screen = pygame.display.set_mode(
            size=(self.layout.width, self.layout.height),
            flags=pygame.RESIZABLE,
        )
        pygame.display.set_caption('Neural Network Visualizer')
        
        # Add these lines to initialize UI controls
        self.ui.update_control_positions(self.layout.width, self.layout.height)
        
        self.network_spec = network_spec
        self.num_steps = num_steps
        self.get_step_data = get_step_data
        
        # Calculate neuron positions and initialize view
        self.neuron_positions = self._calculate_neuron_positions()
        self._initialize_view()
        
        # UI state
        self.current_step = 0
        self.last_update = 0
        self.update_interval = 50  # ms between steps when playing
        self.key_repeat_delay = 300  # ms before key starts repeating
        self.key_repeat_interval = 50  # ms between repeats
        self.last_key_time = 0
        self.key_held_since = 0
        
        # Initialize UI with first step data
        first_step = self.get_step_data(0)
        self.ui.initialize_buttons(first_step)
        
        # Initialize graphics with first step data
        self.graphics.weight_range = self.graphics._calculate_weight_range(first_step)

    def _initialize_view(self) -> None:
        """Initialize camera position to center on network"""
        network_center_x = (
            self.neuron_positions[0][0][0] + 
            self.neuron_positions[-1][0][0]
        ) / 2
        network_center_y = sum(
            pos[1] for layer in self.neuron_positions 
            for pos in layer
        ) / sum(len(layer) for layer in self.neuron_positions)
        
        self.graphics.pan_offset = [
            self.layout.width / 2 - network_center_x,
            self.layout.height / 2 - network_center_y,
        ]

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

    def _step_forward(self) -> None:
        """Move one step forward"""
        self.current_step = min(self.current_step + 1, self.num_steps - 1)
        self.ui.update_step(self.current_step)

    def _step_backward(self) -> None:
        """Move one step backward"""
        self.current_step = max(self.current_step - 1, 0)
        self.ui.update_step(self.current_step)

    def run(self) -> None:
        """Main visualization loop"""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            current_time = pygame.time.get_ticks()
            step_data = self.get_step_data(self.current_step)
            
            # Reset hover states
            self.graphics.hovered_weight = None
            self.graphics.hovered_neuron = None
            
            # Get current mouse position and find hovered elements
            mouse_pos = pygame.mouse.get_pos()
            self._find_hovered_elements(mouse_pos, step_data)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(
                        size=(event.w, event.h),
                        flags=pygame.RESIZABLE,
                    )
                    self.ui.update_control_positions(event.w, event.h)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.ui.is_playing = not self.ui.is_playing
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
                    if event.button == 3:  # Right click
                        self.graphics.is_panning = True
                        self.graphics.last_mouse_pos = event.pos
                    elif event.button in (4, 5):  # Mouse wheel
                        self.graphics._handle_zoom(event.pos, event.button == 4)
                    elif event.button == 1:  # Left click
                        self.ui.handle_ui_click(event.pos, self.num_steps)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:  # Right click release
                        self.graphics.is_panning = False
                    elif event.button == 1:  # Left click release
                        self.ui.is_dragging_slider = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.graphics.is_panning and self.graphics.last_mouse_pos:
                        delta_x = event.pos[0] - self.graphics.last_mouse_pos[0]
                        delta_y = event.pos[1] - self.graphics.last_mouse_pos[1]
                        self.graphics.pan_offset[0] += delta_x
                        self.graphics.pan_offset[1] += delta_y
                        self.graphics.last_mouse_pos = event.pos
                    elif self.ui.is_dragging_slider:
                        self.current_step = self.ui.get_current_step_from_slider(event.pos, self.num_steps)
            
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
            if self.ui.is_playing and current_time - self.last_update > self.update_interval:
                if self.current_step < self.num_steps - 1:
                    self.current_step += 1
                else:
                    self.ui.is_playing = False
                self.last_update = current_time
            
            # Draw
            self.screen.fill(self.graphics.colors['background'])
            self.graphics.draw_network(
                self.screen, 
                step_data,
                self.neuron_positions,
                self.layout,
                show_weights=self.ui.get_button_state('Toggle Weights'),
                show_inputs=self.ui.get_button_state('Toggle Inputs'),
                show_activations=self.ui.get_button_state('Toggle Activations')
            )
            self.ui.draw_controls(self.screen, self.current_step, self.num_steps)
            self.ui.draw_buttons(self.screen)
            self.graphics.draw_hover_info(self.screen)
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

    def _find_hovered_elements(self, mouse_pos: Tuple[int, int], step_data: StepData) -> None:
        """Find hovered neurons and weights"""
        world_pos = self.graphics.screen_to_world(mouse_pos)
        
        # Check neurons first
        for layer_idx, layer_positions in enumerate(self.neuron_positions):
            for neuron_idx, pos in enumerate(layer_positions):
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
                        self.graphics.hovered_neuron = {
                            'value': value,
                            'pos': pos
                        }
                        return

        # If no neuron is hovered, check weights
        if self.ui.get_button_state('Toggle Weights'):
            for layer_idx, (weights, biases) in enumerate(zip(step_data.weights, step_data.biases)):
                for i, pos1 in enumerate(self.neuron_positions[layer_idx]):
                    for j, pos2 in enumerate(self.neuron_positions[layer_idx + 1]):
                        p1 = pygame.Vector2(pos1)
                        p2 = pygame.Vector2(pos2)
                        p3 = pygame.Vector2(world_pos)
                        
                        line_vec = p2 - p1
                        line_length = line_vec.length()
                        if line_length == 0:
                            continue
                        
                        point_vec = p3 - p1
                        t = max(0, min(1, point_vec.dot(line_vec) / (line_length * line_length)))
                        closest = p1 + line_vec * t
                        
                        distance = (p3 - closest).length()
                        if distance < 5 / self.graphics.zoom:
                            self.graphics.hovered_weight = {
                                'value': weights[j, i],
                                'pos': (closest.x, closest.y)
                            }
                            return


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
