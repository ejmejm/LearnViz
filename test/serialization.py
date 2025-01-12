import numpy as np
import pytest
import os
from typing import List, Tuple

from learnviz.network_structure import LayerSpec, LayerType
from learnviz.serialization import LearningDataSerializer


@pytest.fixture
def simple_network() -> List[LayerSpec]:
    """Create a simple 2-layer network specification."""
    return [
        LayerSpec(LayerType.LINEAR, 4, 8),
        LayerSpec(LayerType.LINEAR, 8, 2)
    ]


@pytest.fixture
def sample_weights(simple_network) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create sample weights and biases for the simple network."""
    weights = [
        np.random.randn(8, 4),  # layer 1 weights
        np.random.randn(2, 8)   # layer 2 weights
    ]
    biases = [
        np.random.randn(8),     # layer 1 bias
        np.random.randn(2)      # layer 2 bias
    ]
    return weights, biases


def test_basic_initialization(tmp_path, simple_network):
    """Test basic initialization of the serializer."""
    file_path = tmp_path / 'test.bin'
    
    serializer = LearningDataSerializer()
    serializer.initialize(str(file_path), simple_network)
    serializer.close()
    
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


def test_serialize_weights(tmp_path, simple_network, sample_weights):
    """Test serializing weights and biases."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    serializer = LearningDataSerializer()
    serializer.initialize(str(file_path), simple_network)
    
    # Should work without error
    serializer.serialize_step(weights=weights, biases=biases)
    serializer.close()


def test_serialize_with_extra_values(tmp_path, simple_network, sample_weights):
    """Test serializing with extra per-weight and scalar values."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    # Create learning rates matching weight shapes
    learning_rates = [
        (np.full_like(w, 0.01), np.full_like(b, 0.01))
        for w, b in zip(weights, biases)
    ]
    
    serializer = LearningDataSerializer()
    serializer.initialize(
        str(file_path),
        simple_network,
        per_weight_values={'learning_rate': np.float32},
        extra_values={'global_step': np.int32}
    )
    
    # Should work without error
    serializer.serialize_step(
        weights=weights,
        biases=biases,
        per_weight_values={'learning_rate': learning_rates},
        extra_values={'global_step': 0}
    )
    serializer.close()


def test_serialize_with_all_options(tmp_path, simple_network, sample_weights):
    """Test serializing with all optional data enabled."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    serializer = LearningDataSerializer()
    serializer.initialize(
        str(file_path),
        simple_network,
        store_inputs=True,
        store_activations=True,
        store_predictions=True,
        store_loss=True
    )
    
    # Create sample data
    inputs = np.random.randn(4)  # batch_size=32, input_size=4
    activations = [
        np.random.randn(8),  # layer 1 output
        np.random.randn(2)   # layer 2 output
    ]
    predictions = activations[-1]
    loss = 0.5
    
    # Should work without error
    serializer.serialize_step(
        weights=weights,
        biases=biases,
        inputs=inputs,
        activations=activations,
        predictions=predictions,
        loss=loss
    )
    serializer.close()


def test_validation_errors(tmp_path, simple_network, sample_weights):
    """Test that validation errors are raised for incorrect data."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    serializer = LearningDataSerializer()
    serializer.initialize(str(file_path), simple_network)
    
    # Test wrong weight shape
    wrong_weights = [np.random.randn(2, 2) for _ in range(2)]
    with pytest.raises(ValueError, match='Invalid shape'):
        serializer.serialize_step(weights=wrong_weights, biases=biases)
    
    # Test missing bias
    with pytest.raises(ValueError, match='Missing bias'):
        serializer.serialize_step(weights=weights, biases=biases[:-1])
    
    serializer.close()


def test_half_precision(tmp_path, simple_network, sample_weights):
    """Test serialization with half precision."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    serializer = LearningDataSerializer()
    serializer.initialize(str(file_path), simple_network, half_precision=True)
    
    # Should convert to float16
    serializer.serialize_step(weights=weights, biases=biases)
    serializer.close()
