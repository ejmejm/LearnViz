def test_deserialize_step():
    """Test deserializing a single step."""
    serializer = LearningDataSerializer()
    deserializer = LearningDataDeserializer()
    
    with tempfile.NamedTemporaryFile() as f:
        # Initialize with test network
        network_spec = NetworkSpec([
            LayerSpec(LayerType.LINEAR, 2, 3, True),
            LayerSpec(LayerType.LINEAR, 3, 1, True)
        ])
        
        serializer.initialize(f.name, network_spec)
        
        # Write test data
        weights = [np.array([[1, 2], [3, 4], [5, 6]]), np.array([[7, 8, 9]])]
        biases = [np.array([0.1, 0.2, 0.3]), np.array([0.4])]
        serializer.serialize_step(weights=weights, biases=biases)
        serializer.close()
        
        # Read and verify
        deserializer.initialize(f.name)
        step_data = deserializer.deserialize_step(0)
        
        assert isinstance(step_data, StepData)
        np.testing.assert_array_equal(step_data.weights[0], weights[0])
        np.testing.assert_array_equal(step_data.weights[1], weights[1])
        np.testing.assert_array_equal(step_data.biases[0], biases[0])
        np.testing.assert_array_equal(step_data.biases[1], biases[1])
        assert step_data.inputs is None
        assert step_data.activations is None
        assert step_data.predictions is None
        assert step_data.loss is None
        assert step_data.per_weight_values is None
        assert step_data.extra_values is None


def test_deserialize_with_optional_data():
    """Test deserializing data with optional values enabled."""
    serializer = LearningDataSerializer()
    deserializer = LearningDataDeserializer()
    
    with tempfile.NamedTemporaryFile() as f:
        network_spec = NetworkSpec([
            LayerSpec(LayerType.LINEAR, 2, 3, True),
            LayerSpec(LayerType.LINEAR, 3, 1, True)
        ])
        
        serializer.initialize(
            f.name,
            network_spec,
            store_inputs=True,
            store_activations=True,
            store_predictions=True,
            store_loss=True
        )
        
        # Write test data
        weights = [np.array([[1, 2], [3, 4], [5, 6]]), np.array([[7, 8, 9]])]
        biases = [np.array([0.1, 0.2, 0.3]), np.array([0.4])]
        inputs = np.array([0.5, 0.6])
        activations = [np.array([0.7, 0.8, 0.9]), np.array([1.0])]
        predictions = np.array([1.1])
        loss = 0.5
        
        serializer.serialize_step(
            weights=weights,
            biases=biases,
            inputs=inputs,
            activations=activations,
            predictions=predictions,
            loss=loss
        )
        serializer.close()
        
        # Read and verify
        deserializer.initialize(f.name)
        step_data = deserializer.deserialize_step(0)
        
        assert isinstance(step_data, StepData)
        np.testing.assert_array_equal(step_data.weights[0], weights[0])
        np.testing.assert_array_equal(step_data.weights[1], weights[1])
        np.testing.assert_array_equal(step_data.biases[0], biases[0])
        np.testing.assert_array_equal(step_data.biases[1], biases[1])
        np.testing.assert_array_equal(step_data.inputs, inputs)
        np.testing.assert_array_equal(step_data.activations[0], activations[0])
        np.testing.assert_array_equal(step_data.activations[1], activations[1])
        np.testing.assert_array_equal(step_data.predictions, predictions)
        assert step_data.loss == loss
        assert step_data.per_weight_values is None
        assert step_data.extra_values is None


def test_deserialize_with_custom_values():
    """Test deserializing data with custom per-weight and extra values."""
    serializer = LearningDataSerializer()
    deserializer = LearningDataDeserializer()
    
    with tempfile.NamedTemporaryFile() as f:
        network_spec = NetworkSpec([
            LayerSpec(LayerType.LINEAR, 2, 3, True),
            LayerSpec(LayerType.LINEAR, 3, 1, True)
        ])
        
        per_weight_values = {'gradients': np.float32}
        extra_values = {'learning_rate': np.float32}
        
        serializer.initialize(
            f.name,
            network_spec,
            per_weight_values=per_weight_values,
            extra_values=extra_values
        )
        
        # Write test data
        weights = [np.array([[1, 2], [3, 4], [5, 6]]), np.array([[7, 8, 9]])]
        biases = [np.array([0.1, 0.2, 0.3]), np.array([0.4])]
        gradients = [
            (np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]),
             np.array([0.001, 0.002, 0.003])),
            (np.array([[0.07, 0.08, 0.09]]), np.array([0.004]))
        ]
        learning_rate = 0.01
        
        serializer.serialize_step(
            weights=weights,
            biases=biases,
            per_weight_values={'gradients': gradients},
            extra_values={'learning_rate': learning_rate}
        )
        serializer.close()
        
        # Read and verify
        deserializer.initialize(f.name)
        step_data = deserializer.deserialize_step(0)
        
        assert isinstance(step_data, StepData)
        np.testing.assert_array_equal(step_data.weights[0], weights[0])
        np.testing.assert_array_equal(step_data.weights[1], weights[1])
        np.testing.assert_array_equal(step_data.biases[0], biases[0])
        np.testing.assert_array_equal(step_data.biases[1], biases[1])
        
        grad_data = step_data.per_weight_values['gradients']
        np.testing.assert_array_equal(grad_data[0][0], gradients[0][0])
        np.testing.assert_array_equal(grad_data[0][1], gradients[0][1])
        np.testing.assert_array_equal(grad_data[1][0], gradients[1][0])
        np.testing.assert_array_equal(grad_data[1][1], gradients[1][1])
        
        assert step_data.extra_values['learning_rate'] == learning_rate 