from dataclasses import dataclass
from enum import Enum
import numpy as np
import struct
from typing import Dict, List, Optional, Union, Sequence, Tuple

from learnviz.network_structure import LayerSpec


class LearningDataSerializer:
    def __init__(self):
        self.file = None
        self.header_size = None
        self.step_size = None
        self.store_inputs = False
        self.store_activations = False
        self.store_predictions = False
        self.store_loss = False
        self.extra_values = None
        self.layers = None
        self.precision = None
        self._struct_format = None
    

    def initialize(
        self,
        file_path: str,
        layers: List[LayerSpec],
        store_inputs: bool = False,
        store_activations: bool = False,
        store_predictions: bool = False,
        store_loss: bool = False,
        per_weight_values: Optional[Dict[str, np.dtype]] = None,
        extra_values: Optional[Dict[str, np.dtype]] = None,
        half_precision: bool = False
    ) -> None:
        """Initialize the serializer and create the binary file with header.
        
        Args:
            file_path: Path where the binary file will be created
            layers: List of layer specifications defining the network architecture
            store_inputs: Whether to store input data for each step
            store_activations: Whether to store activation values for each layer
            store_predictions: Whether to store model predictions
            store_loss: Whether to store loss values
            per_weight_values: Dictionary mapping names to numpy dtypes for values that
                             mirror the weight matrices in size
            extra_values: Dictionary mapping names to numpy dtypes for scalar values
            half_precision: Whether to use float16 instead of float32 for weights
        """
        self.file = open(file_path, 'wb')
        self.layers = layers
        self.store_inputs = store_inputs
        self.store_activations = store_activations
        self.store_predictions = store_predictions
        self.store_loss = store_loss
        self.per_weight_values = per_weight_values or {}
        self.extra_values = extra_values or {}
        self.half_precision = half_precision
        
        # Write header...
        self.file.write(b'NNVIZ')
        self.file.write(struct.pack('B', 1))  # Version 1
        
        flags = (store_inputs << 0 | store_activations << 1 |
                store_predictions << 2 | store_loss << 3)
        self.file.write(struct.pack('B', flags))
        
        # Write network structure...
        self.file.write(struct.pack('I', len(layers)))
        for layer in layers:
            self.file.write(struct.pack('BIIB', 
                layer.layer_type.value,
                layer.input_size,
                layer.output_size,
                layer.has_bias
            ))
        
        # Write per-weight values info
        self.file.write(struct.pack('I', len(self.per_weight_values)))
        for name, dtype in self.per_weight_values.items():
            name_bytes = name.encode('utf-8')
            self.file.write(struct.pack('B', len(name_bytes)))
            self.file.write(name_bytes)
            self.file.write(struct.pack('B', self._get_dtype_code(dtype)))
        
        # Write extra values info (now just name and dtype for scalars)
        self.file.write(struct.pack('I', len(self.extra_values)))
        for name, dtype in self.extra_values.items():
            name_bytes = name.encode('utf-8')
            self.file.write(struct.pack('B', len(name_bytes)))
            self.file.write(name_bytes)
            self.file.write(struct.pack('B', self._get_dtype_code(dtype)))
        
        # Write precision flag
        self.file.write(struct.pack('B', half_precision))
        
        # Calculate and write step size
        self.step_size = self._calculate_step_size()
        self.file.write(struct.pack('Q', self.step_size))
        
        # Store header size for later use
        self.header_size = self.file.tell()
        
        # Create struct format for step data
        self._create_step_format()


    def serialize_step(
        self,
        weights: List[np.ndarray],
        biases: Optional[List[np.ndarray]] = None,
        inputs: Optional[np.ndarray] = None,
        activations: Optional[List[np.ndarray]] = None,
        predictions: Optional[np.ndarray] = None,
        loss: Optional[float] = None,
        per_weight_values: Optional[Dict[str, List[Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]]]] = None,
        extra_values: Optional[Dict[str, Union[float, int, bool]]] = None
    ) -> None:
        """Serialize one step of training data."""
        if self.file is None:
            raise RuntimeError("Serializer not initialized")
            
        data = []
        
        # Add weights, biases, and per-weight values for each layer
        for i, layer in enumerate(self.layers):
            # Add weights
            w = self._validate_array(weights[i], 
                                  (layer.output_size, layer.input_size),
                                  f'weights layer {i}')
            data.extend(w.flatten())
            
            # Add biases if present
            if layer.has_bias:
                if biases is None or i >= len(biases):
                    raise ValueError(f'Missing bias for layer {i}')
                b = self._validate_array(biases[i], 
                                      (layer.output_size,),
                                      f'bias layer {i}')
                data.extend(b.flatten())
            
            # Add per-weight values
            if per_weight_values:
                for name, value_list in per_weight_values.items():
                    if name not in self.per_weight_values:
                        raise ValueError(f'Unexpected per-weight value: {name}')
                    
                    layer_values = value_list[i]
                    if isinstance(layer_values, (tuple, list)):
                        w_values, b_values = layer_values
                    else:
                        w_values = layer_values
                        b_values = None
                    
                    v = self._validate_array(w_values,
                                          w.shape,
                                          f'{name} weights layer {i}')
                    data.extend(v.flatten())
                    
                    if layer.has_bias:
                        if b_values is None:
                            raise ValueError(
                                f'Missing bias values for {name} in layer {i}'
                            )
                        v_bias = self._validate_array(b_values,
                                                   b.shape,
                                                   f'{name} bias layer {i}')
                        data.extend(v_bias.flatten())
        
        # Add optional data
        if self.store_inputs:
            if inputs is None:
                raise ValueError('Inputs required but not provided')
            data.extend(inputs.flatten())
            
        if self.store_activations:
            if activations is None:
                raise ValueError('Activations required but not provided')
            for i, act in enumerate(activations):
                data.extend(act.flatten())
                
        if self.store_predictions:
            if predictions is None:
                raise ValueError('Predictions required but not provided')
            data.extend(predictions.flatten())
            
        if self.store_loss:
            if loss is None:
                raise ValueError('Loss required but not provided')
            data.append(loss)
        
        # Add extra scalar values
        if self.extra_values:
            if extra_values is None:
                raise ValueError('Extra values required but not provided')
            for name in self.extra_values:
                if name not in extra_values:
                    raise ValueError(f'Missing extra value: {name}')
                data.append(extra_values[name])
        
        # Write the data
        self.file.write(struct.pack(self._struct_format, *data))


    def close(self) -> None:
        """Close the binary file."""
        if self.file:
            self.file.close()
            self.file = None


    def _calculate_step_size(self) -> int:
        """Calculate the size in bytes of each step."""
        size = 0
        float_size = 2 if self.half_precision else 4
        
        # Weights and biases
        for layer in self.layers:
            weight_size = layer.input_size * layer.output_size
            size += weight_size * float_size
            if layer.has_bias:
                size += layer.output_size * float_size
            
            # Per-weight values
            for dtype in self.per_weight_values.values():
                dtype_size = np.dtype(dtype).itemsize
                size += weight_size * dtype_size
                if layer.has_bias:
                    size += layer.output_size * dtype_size
        
        # Optional data
        if self.store_inputs:
            size += self.layers[0].input_size * float_size
            
        if self.store_activations:
            for layer in self.layers:
                size += layer.output_size * float_size
                
        if self.store_predictions:
            size += self.layers[-1].output_size * float_size
            
        if self.store_loss:
            size += float_size
            
        # Extra scalar values
        for dtype in self.extra_values.values():
            size += np.dtype(dtype).itemsize
            
        return size


    def _create_step_format(self) -> None:
        """Create the struct format string for step data."""
        format_chars = []
        float_format = 'e' if self.half_precision else 'f'
        
        # Weights, biases, and per-weight values
        for layer in self.layers:
            weight_size = layer.input_size * layer.output_size
            format_chars.extend([float_format] * weight_size)
            
            if layer.has_bias:
                format_chars.extend([float_format] * layer.output_size)
            
            # Per-weight values
            for dtype in self.per_weight_values.values():
                format_chars.extend([self._get_dtype_format(dtype)] * weight_size)
                if layer.has_bias:
                    format_chars.extend([self._get_dtype_format(dtype)] * layer.output_size)
        
        # Optional data
        if self.store_inputs:
            format_chars.extend([float_format] * self.layers[0].input_size)
            
        if self.store_activations:
            for layer in self.layers:
                format_chars.extend([float_format] * layer.output_size)
                
        if self.store_predictions:
            format_chars.extend([float_format] * self.layers[-1].output_size)
            
        if self.store_loss:
            format_chars.append(float_format)
            
        # Extra scalar values
        for dtype in self.extra_values.values():
            format_chars.append(self._get_dtype_format(dtype))
            
        self._struct_format = '<' + ''.join(format_chars)


    def _validate_array(
        self,
        arr: np.ndarray,
        expected_shape: Tuple[int, ...],
        name: str
    ) -> np.ndarray:
        """Validate array shape and convert to correct precision."""
        if arr.shape != expected_shape:
            raise ValueError(
                f'Invalid shape for {name}: expected {expected_shape}, got {arr.shape}'
            )
        dtype = np.float16 if self.precision == 16 else np.float32
        return arr.astype(dtype)


    @staticmethod
    def _get_dtype_code(dtype: np.dtype) -> int:
        """Convert numpy dtype to format code for storage."""
        dtype = np.dtype(dtype)
        if dtype == np.float16:
            return 1
        elif dtype == np.float32:
            return 2
        elif dtype == np.int32:
            return 3
        elif dtype == np.bool_:
            return 4
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')


    @staticmethod
    def _get_dtype_format(dtype: np.dtype) -> str:
        """Convert numpy dtype to struct format character."""
        dtype = np.dtype(dtype)
        if dtype == np.float16:
            return 'e'
        elif dtype == np.float32:
            return 'f'
        elif dtype == np.int32:
            return 'i'
        elif dtype == np.bool_:
            return '?'
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
