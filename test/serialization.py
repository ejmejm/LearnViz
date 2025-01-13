import numpy as np
import pytest
from typing import List, Tuple

from learnviz.network_structure import LayerSpec, LayerType, NetworkSpec
from learnviz.serialization import LearningDataSerializer, LearningDataDeserializer


@pytest.fixture
def simple_network() -> List[LayerSpec]:
    """Create a simple 2-layer network specification."""
    return NetworkSpec([
        LayerSpec(LayerType.LINEAR, 4, 8),
        LayerSpec(LayerType.LINEAR, 8, 2)
    ])


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
    """Test basic initialization of the serializer and deserializer."""
    file_path = tmp_path / 'test.bin'
    
    # Serialize
    serializer = LearningDataSerializer()
    serializer.initialize(str(file_path), simple_network)
    serializer.close()
    
    # Deserialize
    deserializer = LearningDataDeserializer()
    deserializer.initialize(str(file_path))
    
    # Check network structure matches
    assert len(deserializer.layers) == len(simple_network.layers)
    for orig, loaded in zip(simple_network.layers, deserializer.layers):
        assert orig.layer_type == loaded.layer_type
        assert orig.input_size == loaded.input_size
        assert orig.output_size == loaded.output_size
        assert orig.has_bias == loaded.has_bias
    
    deserializer.close()


def test_serialize_weights(tmp_path, simple_network, sample_weights):
    """Test serializing and deserializing weights and biases."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    # Serialize
    serializer = LearningDataSerializer()
    serializer.initialize(str(file_path), simple_network)
    serializer.serialize_step(weights=weights, biases=biases)
    serializer.close()
    
    # Deserialize
    deserializer = LearningDataDeserializer()
    deserializer.initialize(str(file_path))
    data = deserializer.deserialize_step(0)
    deserializer.close()
    
    # Check data matches
    assert len(data.weights) == len(weights)
    assert len(data.biases) == len(biases)
    for w1, w2 in zip(weights, data.weights):
        np.testing.assert_allclose(w1, w2)
    for b1, b2 in zip(biases, data.biases):
        np.testing.assert_allclose(b1, b2)


def test_serialize_with_extra_values(tmp_path, simple_network, sample_weights):
    """Test serializing and deserializing with extra per-weight and scalar values."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    # Create learning rates matching weight shapes
    learning_rates = [
        (np.full_like(w, 0.01), np.full_like(b, 0.01))
        for w, b in zip(weights, biases)
    ]
    
    # Serialize
    serializer = LearningDataSerializer()
    serializer.initialize(
        str(file_path),
        simple_network,
        per_weight_values={'learning_rate': np.float32},
        extra_values={'global_step': np.int32}
    )
    
    serializer.serialize_step(
        weights=weights,
        biases=biases,
        per_weight_values={'learning_rate': learning_rates},
        extra_values={'global_step': 42}
    )
    serializer.close()
    
    # Deserialize
    deserializer = LearningDataDeserializer()
    deserializer.initialize(str(file_path))
    data = deserializer.deserialize_step(0)
    deserializer.close()
    
    # Check extra values match
    assert data.extra_values['global_step'] == 42
    for i, (w_lr, b_lr) in enumerate(learning_rates):
        np.testing.assert_allclose(w_lr, data.per_weight_values['learning_rate'][i][0])
        np.testing.assert_allclose(b_lr, data.per_weight_values['learning_rate'][i][1])


def test_serialize_with_all_options(tmp_path, simple_network, sample_weights):
    """Test serializing and deserializing with all optional data enabled."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    # Create sample data
    inputs = np.random.randn(4)
    activations = [
        np.random.randn(8),  # layer 1 output
        np.random.randn(2)   # layer 2 output
    ]
    predictions = activations[-1]
    loss = 0.5
    
    # Serialize
    serializer = LearningDataSerializer()
    serializer.initialize(
        str(file_path),
        simple_network,
        store_inputs=True,
        store_activations=True,
        store_predictions=True,
        store_loss=True
    )
    
    serializer.serialize_step(
        weights=weights,
        biases=biases,
        inputs=inputs,
        activations=activations,
        predictions=predictions,
        loss=loss
    )
    serializer.close()
    
    # Deserialize
    deserializer = LearningDataDeserializer()
    deserializer.initialize(str(file_path))
    data = deserializer.deserialize_step(0)
    deserializer.close()
    
    # Check all data matches
    np.testing.assert_allclose(inputs, data.inputs)
    for act1, act2 in zip(activations, data.activations):
        np.testing.assert_allclose(act1, act2)
    np.testing.assert_allclose(predictions, data.predictions)
    assert loss == data.loss


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
    """Test serialization and deserialization with half precision."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    # Serialize
    serializer = LearningDataSerializer()
    serializer.initialize(str(file_path), simple_network, half_precision=True)
    serializer.serialize_step(weights=weights, biases=biases)
    serializer.close()
    
    # Deserialize
    deserializer = LearningDataDeserializer()
    deserializer.initialize(str(file_path))
    data = deserializer.deserialize_step(0)
    deserializer.close()
    
    # Check data matches with reduced precision
    assert data.weights[0].dtype == np.float16
    assert data.biases[0].dtype == np.float16
    for w1, w2 in zip(weights, data.weights):
        np.testing.assert_allclose(w1, w2, rtol=1e-3)
    for b1, b2 in zip(biases, data.biases):
        np.testing.assert_allclose(b1, b2, rtol=1e-3)


def test_multiple_steps(tmp_path, simple_network, sample_weights):
    """Test serializing and deserializing multiple steps."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    # Serialize multiple steps
    serializer = LearningDataSerializer()
    serializer.initialize(str(file_path), simple_network)
    
    num_steps = 3
    for step in range(num_steps):
        # Modify weights slightly for each step
        step_weights = [w + step * 0.1 for w in weights]
        step_biases = [b + step * 0.1 for b in biases]
        serializer.serialize_step(weights=step_weights, biases=step_biases)
    
    serializer.close()
    
    # Deserialize and check
    deserializer = LearningDataDeserializer()
    deserializer.initialize(str(file_path))
    
    assert deserializer.total_steps == num_steps
    
    # Test individual step access
    for step in range(num_steps):
        data = deserializer.deserialize_step(step)
        expected_weights = [w + step * 0.1 for w in weights]
        expected_biases = [b + step * 0.1 for b in biases]
        
        for w1, w2 in zip(expected_weights, data.weights):
            np.testing.assert_allclose(w1, w2)
        for b1, b2 in zip(expected_biases, data.biases):
            np.testing.assert_allclose(b1, b2)
    
    deserializer.close()


def test_step_iterator(tmp_path, simple_network, sample_weights):
    """Test iterating over steps."""
    file_path = tmp_path / 'test.bin'
    weights, biases = sample_weights
    
    # Serialize multiple steps
    serializer = LearningDataSerializer()
    serializer.initialize(str(file_path), simple_network)
    
    num_steps = 3
    for step in range(num_steps):
        step_weights = [w + step * 0.1 for w in weights]
        step_biases = [b + step * 0.1 for b in biases]
        serializer.serialize_step(weights=step_weights, biases=step_biases)
    
    serializer.close()
    
    # Test iteration
    deserializer = LearningDataDeserializer()
    deserializer.initialize(str(file_path))
    
    for step, data in enumerate(deserializer.iter_steps()):
        expected_weights = [w + step * 0.1 for w in weights]
        expected_biases = [b + step * 0.1 for b in biases]
        
        for w1, w2 in zip(expected_weights, data.weights):
            np.testing.assert_allclose(w1, w2)
        for b1, b2 in zip(expected_biases, data.biases):
            np.testing.assert_allclose(b1, b2)
    
    deserializer.close()
