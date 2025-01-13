from dataclasses import dataclass
import pygame
from typing import Dict, Tuple, Optional, Any
from ..serialization import StepData


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


class UIManager:
    """Handles UI-related functionality"""
    def __init__(self, layout: LayoutConfig):
        self.layout = layout
        self.buttons: Dict[str, Button] = {}
        self.is_playing = False
        self.is_dragging_slider = False
        self.play_pause_button_rect: Optional[pygame.Rect] = None
        self.slider_rect: Optional[pygame.Rect] = None
        self.colors = self._create_color_scheme()
        self.font = pygame.font.Font(None, self.layout.button_text_size)
        self.current_step = 0
        self.last_update = 0
        self.update_interval = 50

    def _create_color_scheme(self) -> Dict[str, Tuple[int, int, int]]:
        """Create the default color scheme for UI elements"""
        return {
            'ui': (200, 200, 200),
            'slider_progress': (255, 0, 0),
            'button_active': (240, 240, 240),
            'button_inactive': (100, 100, 100),
            'button_text_active': (50, 50, 50),
            'button_text_inactive': (240, 240, 240),
        }

    def initialize_buttons(self, step_data: StepData) -> None:
        """Initialize toggle buttons based on available data"""
        button_names = ['Toggle Weights']
        
        if step_data.inputs is not None:
            button_names.append('Toggle Inputs')
        if step_data.activations is not None:
            button_names.append('Toggle Activations')
        if step_data.loss is not None:
            button_names.append('Toggle Loss')
            
        if step_data.per_weight_values:
            for key in step_data.per_weight_values:
                button_name = f"Toggle {key.replace('_', ' ').title()}"
                button_names.append(button_name)
                
        if step_data.extra_values:
            for key in step_data.extra_values:
                button_name = f"Toggle {key.replace('_', ' ').title()}"
                button_names.append(button_name)
        
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

    def update_control_positions(self, screen_width: int, screen_height: int) -> None:
        """Update UI control positions based on current screen size"""
        play_button_width = 40
        slider_width = screen_width - 400
        total_width = play_button_width + 10 + slider_width
        
        start_x = (screen_width - total_width) // 2
        play_button_y = screen_height - 50
        
        self.play_pause_button_rect = pygame.Rect(
            start_x,
            play_button_y,
            play_button_width,
            40,
        )
        
        slider_y = play_button_y + (self.play_pause_button_rect.height // 2) - 2
        self.slider_rect = pygame.Rect(
            start_x + play_button_width + 10,
            slider_y,
            slider_width,
            4,
        )

    def handle_ui_click(self, mouse_pos: Tuple[int, int], num_steps: int) -> None:
        """Handle clicks on UI elements"""
        for button in self.buttons.values():
            if button.rect.collidepoint(mouse_pos):
                button.is_active = not button.is_active
                return
                
        if (self.slider_rect and 
            abs(mouse_pos[1] - self.slider_rect.centery) <= 8 and 
            self.slider_rect.left <= mouse_pos[0] <= self.slider_rect.right):
            self.is_dragging_slider = True
            
        elif self.play_pause_button_rect and self.play_pause_button_rect.collidepoint(mouse_pos):
            self.is_playing = not self.is_playing

    def draw_controls(self, screen: pygame.Surface, current_step: int, num_steps: int) -> None:
        """Draw the UI controls (play/pause button and slider)"""
        if not (self.slider_rect and self.play_pause_button_rect):
            return
            
        progress_width = (current_step / num_steps) * self.slider_rect.width
        
        # Draw progress part of slider
        progress_rect = pygame.Rect(
            self.slider_rect.left,
            self.slider_rect.top,
            progress_width,
            self.slider_rect.height
        )
        pygame.draw.rect(screen, self.colors['slider_progress'], progress_rect)
        
        # Draw remaining part of slider
        remaining_rect = pygame.Rect(
            self.slider_rect.left + progress_width,
            self.slider_rect.top,
            self.slider_rect.width - progress_width,
            self.slider_rect.height
        )
        pygame.draw.rect(screen, self.colors['ui'], remaining_rect)
        
        # Draw slider handle
        handle_pos = (
            self.slider_rect.left + progress_width,
            self.slider_rect.centery
        )
        pygame.draw.circle(screen, self.colors['slider_progress'], handle_pos, 8)
        
        # Draw play/pause icon
        if not self.is_playing:
            # Play triangle
            pygame.draw.polygon(
                screen,
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
                screen,
                self.colors['ui'],
                (self.play_pause_button_rect.left + 10, self.play_pause_button_rect.top + 10, 5, 20)
            )
            pygame.draw.rect(
                screen,
                self.colors['ui'],
                (self.play_pause_button_rect.right - 15, self.play_pause_button_rect.top + 10, 5, 20)
            )

    def draw_buttons(self, screen: pygame.Surface) -> None:
        """Draw all toggle buttons"""
        for button in self.buttons.values():
            # Draw button background
            color = self.colors['button_active'] if button.is_active else self.colors['button_inactive']
            pygame.draw.rect(screen, color, button.rect)
            
            # Draw button text
            text_color = self.colors['button_text_active'] if button.is_active else self.colors['button_text_inactive']
            text_surface = self.font.render(button.text, True, text_color)
            text_rect = text_surface.get_rect(center=button.rect.center)
            screen.blit(text_surface, text_rect)

    def get_current_step_from_slider(self, mouse_pos: Tuple[int, int], num_steps: int) -> int:
        """Calculate current step based on slider position"""
        if not self.slider_rect:
            return self.current_step
            
        rel_x = (mouse_pos[0] - self.slider_rect.left) / self.slider_rect.width
        rel_x = max(0, min(1, rel_x))
        step = int(rel_x * num_steps)
        self.current_step = min(step, num_steps - 1)
        return self.current_step

    def get_button_state(self, name: str) -> bool:
        """Get the state of a button by name"""
        if name in self.buttons:
            return self.buttons[name].is_active
        return False 

    def update_step(self, step: int) -> None:
        """Update the current step"""
        self.current_step = step 