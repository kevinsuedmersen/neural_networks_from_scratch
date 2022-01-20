import copy
import itertools

import numpy as np
import pytest

import src.constants as c
from src.data_generators.factory import get_data_generator
from src.model_architectures import get_model
from tests.test_config import TestConfig


class TestBackwardPropagation(TestConfig):
    """Tests that the backpropagation algorithm computes the correct gradients, i.e. the same gradients
    which are computed using a brute force method
    """
    @pytest.fixture(scope="class")
    def img_gen_train(self, config_parser):
        img_gen = get_data_generator(
            config_parser.data_gen_name,
            config_parser.data_dir,
            config_parser.val_size,
            config_parser.test_size,
            config_parser.batch_size,
            config_parser.img_height,
            config_parser.img_width,
            c.RANDOM_STATE
        )
        img_gen_train, n_samples_train = img_gen.train()

        return img_gen_train

    @pytest.fixture(scope="class")
    def single_training_tuple(self, img_gen_train):
        # Retrieve the first image array and first ground truth labels
        x_train, ytrue_train = next(img_gen_train)
        x_train_0 = x_train[0]
        ytrue_train_0 = ytrue_train[0]

        # Add back batch dimension to keep dimensions compatible
        x_train_0 = x_train_0[np.newaxis, ...]
        ytrue_train_0 = ytrue_train_0[np.newaxis, ...]

        return x_train_0, ytrue_train_0

    @pytest.fixture(scope="class")
    def untrained_model(self, config_parser):
        """Untrained model instance"""
        model = get_model(
            model_name=config_parser.model_name,
            img_height=config_parser.img_height,
            img_width=config_parser.img_width,
            n_color_channels=config_parser.n_color_channels,
            random_state=c.RANDOM_STATE
        )
        return model

    @pytest.fixture(scope="class")
    def trained_model_backprop(self, img_gen_train, config_parser, untrained_model, single_training_tuple):
        """Model trained on a single image for 1 epoch using backpropagation"""
        # Create a deepcopy of untrained model so that untrained_model remains untrained
        model = copy.deepcopy(untrained_model)

        # Run the forward and backward pass on that single image
        x_train_0, ytrue_train_0 = single_training_tuple
        model.train_step(x_train_0, ytrue_train_0)

        return model

    @staticmethod
    def _compute_gradients_backprop(trained_model_backprop, gradient_type="weight_gradients"):
        """Computes the weight or bias gradients using backpropagation"""
        # Extract all gradients and store them in a flattened numpy array
        # Flattened, because each layer's weight matrices / bias vectors have different dimensions
        gradients = []
        for layer in trained_model_backprop.layers[1:]:
            flat_gradients = getattr(layer, gradient_type).ravel().reshape(-1, 1)
            gradients.append(flat_gradients)

        # shape=(n_layers, n_neurons, n_neurons_prev)
        gradients_backprop = np.concatenate(gradients, axis=0)

        return gradients_backprop

    @staticmethod
    def _compute_loss(untrained_model, x_train_0, ytrue_train_0):
        """Computes the loss with unchanged parameters"""
        activations_out, dendritic_potentials_out = untrained_model._forward_pass(x_train_0)
        loss = untrained_model.loss.compute_losses(ytrue_train_0, activations_out)

        return loss.item()

    @pytest.fixture
    def trained_model_brute_force(
            self,
            untrained_model,
            single_training_tuple,
            epsilon=1e-6
    ):
        """Computes the weight or bias gradients using a brute force method. Each derivative is
        calculated by computing the losses after slightly changing one parameter each time while
        keeping all other parameters constant, then subtracting the loss value computed with the
        constant parameters and then dividing that difference by the slight change, i.e.:
        (L(slightly_changed_parameters, all_other_parameters) - L(original_parameters)) / slight_change
        """
        # Compute the loss with unchange parameters once
        x_train_0, ytrue_train_0 = single_training_tuple
        loss_unchanged_parameters = self._compute_loss(untrained_model, x_train_0, ytrue_train_0)

        # Change each parameter slightly, re-compute the loss and calculate the partial derivatives
        for l in range(1, untrained_model.n_layers):
            biases_shape = untrained_model.layers[l].biases.shape
            weights_shape = untrained_model.layers[l].weights.shape
            for row_idx, col_idx in itertools.product(weights_shape[1]), range(weights_shape[2]):
                # Make sure the next time we change a parameter, we keep all other parameters unchanged
                trained_model_ = copy.deepcopy(untrained_model)

                # Slightly change the parameter value
                slightly_changed_value = trained_model_.weights[:, row_idx, col_idx] + epsilon
                trained_model_.weights[:, row_idx, col_idx] = slightly_changed_value

                # Compute loss with changed parameters
                loss_changed_parameters = self._compute_loss(trained_model_, x_train_0, ytrue_train_0)

                # Compute and set partial derivative
                partial_derivative = (loss_changed_parameters - loss_unchanged_parameters) / epsilon
                trained_model_.weight_gradients[:, row_idx, col_idx] = partial_derivative

        return trained_model_

    def _compare_gradients(self, trained_model_backprop, trained_model_brute_force, gradient_type="weight_gradients"):
        """Iterates through each layer of both networks and compares gradients"""
        pass

    @pytest.fixture
    def mean_absolute_percentage_error(self):
        pass

    def test_weight_gradients(self, trained_model_backprop, config_parser, untrained_model, mean_absolute_percentage_error):
        """Tests that the backward propagation algorithm computes the correct weight gradients"""
        weight_gradients_backprop = self._compute_gradients_backprop(trained_model_backprop, "weight_gradients")
        weight_gradients_brute_force = self.trained_model_brute_force(trained_model_backprop, "weight_gradients")
        np.testing.assert_allclose(weight_gradients_backprop, weight_gradients_brute_force)
        mean_absolute_percentage_error = mean_absolute_percentage_error.result(
            weight_gradients_backprop,
            weight_gradients_brute_force
        )

    def test_bias_gradients(self, trained_model_backprop, mean_absolute_percentage_error):
        """Tests that the backward propagation algorithm computes the correct bias gradients"""
        bias_gradients_backprop = self._compute_gradients_backprop(trained_model_backprop, "bias_gradients")
        bias_gradients_brute_force = self.trained_model_brute_force(trained_model_backprop, "bias_gradients")
        np.testing.assert_allclose(bias_gradients_backprop, bias_gradients_brute_force)
        mean_absolute_percentage_error = mean_absolute_percentage_error.result(
            bias_gradients_backprop,
            bias_gradients_brute_force
        )
