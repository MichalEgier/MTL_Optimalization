import numpy as np
import matplotlib.pyplot as plt
import math
import time
#dataset
from keras.datasets import mnist

class Neural_Network:
    def __init__(self,
                 hidden_layer_activation_function,
                 hidden_layer_activation_function_derivative,
                 layers_sizes=(784, 90, 10),
                 #learning_coeff=0.0001,    #dostosowany dla adadelty
                 learning_coeff=0.00001,    #dostosowany dla adama
                 weights_mean=0.0, weights_sigma=0.1, biases_mean=0.0, biases_sigma=0.1,
                 batch_size=25,
                 momentum_coefficient = 0.7,
                 learning_coeffs_mode = "none",
                 epsilon = 0.00000001,
                 gamma = 0.6,
                 name="default",
                 initialize_method="simple",
                 learning_mode="normal"
                 ):

        self.learning_coeff = learning_coeff
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.hidden_layer_activation_function_derivative = hidden_layer_activation_function_derivative
        self.output_layer_activation_function = softmax
        self.batch_size = batch_size
        self.momentum_coefficient = momentum_coefficient
        self.learning_coeffs_mode = learning_coeffs_mode
        self.epsilon = epsilon  #for adagrad, adadelta, adam
        self.gamma  = gamma #for adadelta
        self.initialize_method = initialize_method
        if learning_coeffs_mode == "adagrad" or learning_coeffs_mode == "adadelta" or learning_coeffs_mode == "adam":
            self.weight_gradients = []  #for adagrad, adadelta, adam
            for i in range(len(layers_sizes) - 1):
                self.weight_gradients.append([])
            self.bias_gradients = []    #for adagrad, adadelta, adam
            for i in range(len(layers_sizes) - 1):
                self.bias_gradients.append([])

        self.initialize_weights(weights_mean, weights_sigma, layers_sizes)
        self.initialize_biases(biases_mean, biases_sigma, layers_sizes)

        self.name = name
        self.learning_mode = learning_mode

        if learning_coeffs_mode == "adadelta":
            self.adadeltaActualisator = AdadeltaCoefficientActualiser(self.weights, self.biases)
        elif learning_coeffs_mode == "adam":
            self.adamActualisator = AdamActualisator(self.weights, self.biases)


                    #tu jakos ten adagrad zrefaktoryzowac
    def initialize_weights(self, weights_mean, weights_sigma, layers_sizes):
        self.weights = []
        if self.learning_coeffs_mode == "adagrad":
            self.weights_learning_coefficients = []
            self.weights_learning_coefficients_gradient_squares = []
        if self.learning_coeffs_mode == "adadelta":
            self.weights_learning_coefficients = []
            self.weights_last_rms_arr = []
            self.weights_last_G_E_arr = []
            self.weights_last_theta_E_arr = []
        for i in range(len(layers_sizes) - 1):
            if self.initialize_method == "xavier":
                self.initialize_weights_method_xavier(weights_mean, layers_sizes, i)
            elif self.initialize_method == "ha":
                self.initialize_weights_method_ha(weights_mean, layers_sizes, i)
            else:
                self.initialize_weights_method_simple(weights_mean, weights_sigma, layers_sizes, i)
            if self.learning_coeffs_mode == "adagrad":
                self.weights_learning_coefficients_gradient_squares.append(np.ones((layers_sizes[i+1], layers_sizes[i])))
                self.weights_learning_coefficients.append(np.ones((layers_sizes[i+1], layers_sizes[i])) * 0.0001) #/ math.sqrt(self.epsilon))

    def initialize_weights_method_simple(self, weights_mean, weights_sigma, layers_sizes, i):
        self.weights.append(np.random.normal(weights_mean, weights_sigma, size=(layers_sizes[i + 1], layers_sizes[i])))

    def initialize_weights_method_xavier(self, weights_mean, layers_sizes, i):
        weights_sigma = math.sqrt(2/(layers_sizes[i]+layers_sizes[i+1]))
        self.weights.append(np.random.normal(weights_mean, weights_sigma, size=(layers_sizes[i + 1], layers_sizes[i])))

    def initialize_weights_method_ha(self, weights_mean, layers_sizes, i):
        weights_sigma = math.sqrt(2/(layers_sizes[i]))
        self.weights.append(np.random.normal(weights_mean, weights_sigma, size=(layers_sizes[i + 1], layers_sizes[i])))

    def initialize_biases(self, biases_mean, biases_sigma, layers_sizes):
        self.biases = []
        if self.learning_coeffs_mode == "adagrad":
            self.biases_learning_coefficients = []
            self.biases_learning_coefficients_gradient_squares = []
        for i in range(len(layers_sizes) - 1):
            if self.initialize_method == "xavier":
                self.initialize_biases_method_xavier(biases_mean, layers_sizes, i)
            elif self.initialize_method == "ha":
                self.initialize_biases_method_ha(biases_mean, layers_sizes, i)
            else:
                self.initialize_biases_method_simple(biases_mean, biases_sigma, layers_sizes, i)
            if self.learning_coeffs_mode == "adagrad":
                self.biases_learning_coefficients_gradient_squares.append(np.ones((layers_sizes[i+1], 1)))
                #self.biases_learning_coefficients.append(np.ones((layers_sizes[i+1], 1)) * self.learning_coeff / math.sqrt(self.epsilon)) #/ math.sqrt(self.epsilon))
                self.biases_learning_coefficients.append(np.ones((layers_sizes[i+1], 1)) * 0.0001) #/ math.sqrt(self.epsilon))
                #self.biases_learning_coefficients.append(np.ones((layers_sizes[i+1], 1)) * self.learning_coeff / math.sqrt(self.weight_gradients + self.epsilon)) #/ math.sqrt(self.epsilon))

    def initialize_biases_method_simple(self, biases_mean, biases_sigma, layers_sizes, i):
        self.biases.append(np.random.normal(biases_mean, biases_sigma, size=(layers_sizes[i + 1], 1)))

    def initialize_biases_method_xavier(self, biases_mean, layers_sizes, i):
        biases_sigma = math.sqrt(2/(layers_sizes[i]+layers_sizes[i+1]))
        self.biases.append(np.random.normal(biases_mean, biases_sigma, size=(layers_sizes[i + 1], 1)))

    def initialize_biases_method_ha(self, biases_mean, layers_sizes, i):
        biases_sigma = math.sqrt(2/(layers_sizes[i]))
        self.biases.append(np.random.normal(biases_mean, biases_sigma, size=(layers_sizes[i + 1], 1)))

    def predict(self, X):
        y_predicted, _, _ = self.process_batch(X)
        for x in y_predicted:
            maximal_output_index = np.argmax(x)
            for i in range(10):
                x[i] = 1 if i == maximal_output_index else 0
        return y_predicted

    def process_batch(self, X):
        z_vectors, a_vectors = [], [X.T]
        a = X.T
        iterator = 0
        while iterator < len(self.weights):
            w_l = self.weights[iterator]
            b_l = self.biases[iterator]
            activation_function = \
                self.output_layer_activation_function if iterator == len(self.weights) - 1 \
                else self.hidden_layer_activation_function
            z = w_l.dot(a) + b_l
            a = activation_function(z)
            z_vectors.append(z)
            a_vectors.append(a)
            iterator += 1
        return a.T, a_vectors, z_vectors


    def backward_propagation(self, y_predicted, y, a_vectors, z_vectors):

        deltas = self._compute_delta_vectors( y_predicted, y, z_vectors)

        self._actualise_weights(deltas, a_vectors)
        self._actualise_biases(deltas)

    def backward_propagation_momentum(self, y_predicted, y, a_vectors, z_vectors, last_weights_actualisation=None, last_biases_actualisation=None):

        deltas = self._compute_delta_vectors( y_predicted, y, z_vectors)

        actualisation_weights = self._actualise_weights_momentum(deltas, a_vectors, last_weights_actualisation)
        actualisation_biases = self._actualise_biases_momentum(deltas, last_biases_actualisation)

        return actualisation_weights, actualisation_biases  #for next iterations

    def backward_propagation_nesterow_momentum(self, y_predicted, y, a_vectors, z_vectors, last_weights_actualisation, last_biases_actualisation):
        #to compute deltas using already updated weights and biases with last actualisations
        if last_weights_actualisation != None:
            preactualise_weights_nesterow_momentum(last_weights_actualisation)
        if last_biases_actualisation != None:
            preactualise_biases_nesterow_momentum(last_biases_actualisation)

        deltas = self._compute_delta_vectors( y_predicted, y, z_vectors)

        actualisation_weights = self._actualise_weights(deltas, a_vectors)
        actualisation_biases = self._actualise_biases(deltas)

        for i in range(len(self.weights)):
            actualisation_weights[i] += last_weights_actualisation[i] * self.momentum_coefficient

        for i in range(len(self.biases)):
            actualisation_biases[i] += last_biases_actualisation[i] * self.momentum_coefficient

        return actualisation_weights, actualisation_biases  #for next iterations

    def _compute_delta_vectors(self, y_predicted, y, z_vectors):
        iterator = len(self.weights) - 2
        deltas = [(y - y_predicted).T]
        while iterator >= 0:
            deltas.append(
                self.weights[iterator + 1].T.dot(deltas[-1])
                * self.hidden_layer_activation_function_derivative(z_vectors[iterator])
            )
            iterator -= 1
        deltas.reverse()
        return deltas


    def _actualise_weights(self, deltas, a_vectors):
        actualisation_deltas = []   #for nesterow momentum
        for i in range(len(self.weights)):
            weight_gradient = deltas[i].dot(a_vectors[i].T)
            if self.learning_coeffs_mode == "adagrad":
                actualisation_delta = - self.weights_learning_coefficients[i]/self.batch_size * weight_gradient
            elif self.learning_coeffs_mode == "adadelta":
                actualisation_delta = - self.adadeltaActualisator.actualiseNetwork(i,"w",self.learning_coeff * weight_gradient/self.batch_size)
            elif self.learning_coeffs_mode == "adam":
                actualisation_delta = - self.adamActualisator.actualiseNetwork(i,"w",self.learning_coeff * weight_gradient/self.batch_size)
            else:
                actualisation_delta = -self.learning_coeff/self.batch_size * weight_gradient
            self.weights[i] -= actualisation_delta
            actualisation_deltas.append(actualisation_delta)
            if self.learning_coeffs_mode == "adagrad" or self.learning_coeffs_mode == "adadelta":
                self.weight_gradients[i] = weight_gradient

        if self.learning_coeffs_mode == "adagrad":
            self._actualise_weights_learning_coefficients()
        return actualisation_deltas

    def _actualise_biases(self, deltas):
        actualisation_deltas = []   #for nesterow momentum
        for i in range(len(self.biases)):
            bias_gradient = np.array([deltas[i].sum(axis=1)]).T
            if self.learning_coeffs_mode == "adagrad":
                actualisation_delta = -self.biases_learning_coefficients[i]/self.batch_size * bias_gradient
            elif self.learning_coeffs_mode == "adadelta":
                actualisation_delta = - self.adadeltaActualisator.actualiseNetwork(i,"b",self.learning_coeff * bias_gradient/self.batch_size)
            elif self.learning_coeffs_mode == "adam":
                actualisation_delta = - self.adamActualisator.actualiseNetwork(i, "b",
                                                                   self.learning_coeff * bias_gradient / self.batch_size)
            else:
                actualisation_delta = -self.learning_coeff/self.batch_size * bias_gradient
            self.biases[i] -= actualisation_delta
            actualisation_deltas.append(actualisation_delta)
            if self.learning_coeffs_mode == "adagrad" or self.learning_coeffs_mode == "adadelta":
                self.bias_gradients[i] = bias_gradient
        if self.learning_coeffs_mode == "adagrad":
            self._actualise_biases_learning_coefficients()
        return actualisation_deltas

    def _actualise_weights_momentum(self, deltas, a_vectors, last_actualisation_value):
        actualisation_deltas = []
        for i in range(len(self.weights)):
            if last_actualisation_value == None:
                actualisation_delta = -self.learning_coeff / self.batch_size * deltas[i].dot(a_vectors[i].T)
            else:
                actualisation_delta = -self.learning_coeff/self.batch_size * deltas[i].dot(a_vectors[i].T) + last_actualisation_value[i] * self.momentum_coefficient
            self.weights[i] -= actualisation_delta
            actualisation_deltas.append(actualisation_delta)
        return actualisation_deltas

    def _actualise_biases_momentum(self, deltas, last_actualisation_value):
        actualisation_deltas = []   #for nesterow momentum
        for i in range(len(self.biases)):
            if last_actualisation_value == None:
                actualisation_delta = -self.learning_coeff/self.batch_size * np.array([deltas[i].sum(axis=1)]).T
            else:
                actualisation_delta = -self.learning_coeff/self.batch_size * np.array([deltas[i].sum(axis=1)]).T + last_actualisation_value[i] * self.momentum_coefficient
            self.biases[i] -= actualisation_delta
            actualisation_deltas.append(actualisation_delta)
        return actualisation_deltas

    def _preactualise_weights_nesterow_momentum(self, last_actualisation_value):
        for i in range(len(self.weights)):
            self.weights[i] -= last_actualisation_value[i] * self.momentum_coefficient

    def _preactualise_biases_nesterow_momentum(self, last_actualisation_value):
        for i in range(len(self.weights)):
            self.biases[i] -= last_actualisation_value[i] * self.momentum_coefficient

    def _actualise_weights_learning_coefficients(self):
        #print("Learning coeffs:")
        #print(str(self.weights_learning_coefficients))

        for i in range(len(self.weights_learning_coefficients)):
            #print()
            if self.learning_coeffs_mode == "adagrad":
                gradients = self.weight_gradients[i]
                self.weights_learning_coefficients_gradient_squares[i] += gradients * gradients
                self.weights_learning_coefficients[i] = self.learning_coeff / (np.sqrt(self.weights_learning_coefficients_gradient_squares[i] + self.epsilon))
            elif self.learning_coeffs_mode == "adadelta":
            #    gradients = self.weight_gradients[i]
            #    e_G2_t = self.gamma * self.weights_last_G_E_arr[i] + (1-self.gamma) * gradients * gradients
            #    #print(str(e_G2_t))
            #    rms_G2_t = np.sqrt(e_G2_t + self.epsilon)
            #    self.weights_learning_coefficients[i] = self.weights_last_rms_arr[i] / rms_G2_t
            #    e_theta2_t = self.gamma * self.weights_last_theta_E_arr[i] - (1-self.gamma) * self.weights_learning_coefficients[i] * gradients * self.weights_learning_coefficients[i] * gradients
            #    rms_theta2_t = np.sqrt(e_theta2_t + self.epsilon)
            #    self.weights_last_G_E_arr[i] = e_G2_t   #to compute next e_G2_t
            #    self.weights_last_theta_E_arr[i] = e_theta2_t   #to compute next e_theta2_t
            #    self.weights_last_rms_arr[i] = rms_theta2_t #to compute next weights_learning_coefficients[i]
                gradient = self.weight_gradients[i]
                E_g = self.gamma * self.weights_last_G_E_arr[i].mean() + (1. - self.gamma) * np.square(gradient)

                delta_w = gradient / np.sqrt(E_g.mean() + self.epsilon)
                E_w = self.gamma * self.weights_last_theta_E_arr[i].mean() + (1 - self.gamma) * np.square(delta_w)

                RMS_w = np.sqrt(E_w)
                RMS_g = np.sqrt(E_g + self.epsilon)
                self.weights_learning_coefficients[i] = RMS_w / RMS_g

                self.weights_last_G_E_arr[i] = E_g
                self.weights_last_theta_E_arr[i] = E_w

                #return output

    def _actualise_biases_learning_coefficients(self):
        for i in range(len(self.biases_learning_coefficients)):
            if self.learning_coeffs_mode == "adagrad":
                gradients = self.bias_gradients[i]
                self.biases_learning_coefficients_gradient_squares[i] += gradients * gradients
                self.biases_learning_coefficients[i] = self.learning_coeff / (np.sqrt(self.biases_learning_coefficients_gradient_squares[i] + self.epsilon))
            elif self.learning_coeffs_mode == "adadelta":
                #gradients = self.bias_gradients[i]  #g_t
                #e_G2_t = self.gamma * self.biases_last_G_E_arr[i] + (1-self.gamma) * gradients * gradients
                #rms_G2_t = np.sqrt(e_G2_t + self.epsilon)
                #self.biases_learning_coefficients[i] = self.biases_last_rms_arr[i] / rms_G2_t
                #e_theta2_t = self.gamma * self.biases_last_theta_E_arr[i] - (1-self.gamma) * self.biases_learning_coefficients[i] * gradients * self.biases_learning_coefficients[i] * gradients
                #rms_theta2_t = np.sqrt(e_theta2_t + self.epsilon)
                #self.biases_last_G_E_arr[i] = e_G2_t
                #self.biases_last_theta_E_arr[i] = e_theta2_t
                #gradient = self.weight_gradients[i]
                #self.biases_last_rms_arr[i] = rms_theta2_t

                gradient = self.bias_gradients[i]
                E_g = self.gamma * self.biases_last_G_E_arr[i].mean() + (1. - self.gamma) * np.square(gradient)

                delta_w = gradient / np.sqrt(E_g.mean() + self.epsilon)
                E_w = self.gamma * self.biases_last_theta_E_arr[i].mean() + (1 - self.gamma) * np.square(delta_w)

                RMS_w = np.sqrt(E_w)
                RMS_g = np.sqrt(E_g + self.epsilon)
                self.biases_learning_coefficients[i] = RMS_w / RMS_g

                self.biases_last_G_E_arr[i] = E_g
                self.biases_last_theta_E_arr[i] = E_w


    def train_network(self, training_set_X, training_set_Y, test_set_X, test_set_Y, number_of_epochs):

        errors = []
        accuraccies_in_training_set = []
        accuraccies_in_test_set= []

        iter = 0

        for epoch in range(number_of_epochs):
            training_X_batches = np.array_split(training_set_X, int(len(training_set_X) / self.batch_size))
            training_Y_batches = np.array_split(training_set_Y, int(len(training_set_X) / self.batch_size))
            for Xi, Yi in zip(training_X_batches, training_Y_batches):
                #print("Next batch = " + str(iter))
                iter += 1
                y_predicted, a_vectors, z_vectors = self.process_batch(Xi)
                self.backward_propagation(y_predicted, Yi, a_vectors, z_vectors)

            training_set_network_prediction = self.predict(training_set_X)
            test_set_network_prediction = self.predict(test_set_X)
            training_set_acc = np.where(training_set_Y == training_set_network_prediction,
                                        training_set_Y, 0).sum() / len(training_set_Y)
            test_set_acc = np.where(test_set_Y == test_set_network_prediction,
                                        test_set_Y, 0).sum() / len(test_set_Y)
            error_in_iteration = -(np.log(self.process_batch(test_set_X)[0]) * test_set_Y).sum(axis=1).mean()
            print("Iteration: " + str(epoch) + " Accuracy on training set: " +
                  str(training_set_acc) + " Accuracy on passed test set: " +
                  str(test_set_acc) + " Error in epoch: " + str(error_in_iteration))

            accuraccies_in_test_set.append(test_set_acc)
            accuraccies_in_training_set.append(training_set_acc)
            errors.append(error_in_iteration)

        return accuraccies_in_training_set, accuraccies_in_test_set, errors

    def train_network_momentum(self, training_set_X, training_set_Y, test_set_X, test_set_Y, number_of_epochs):

        if(self.learning_coeffs_mode == "adadelta" or self.learning_coeffs_mode == "adagrad" or self.learning_coeffs_mode == "adam"):
            raise Exception("Momentum is not supported along with weights optimilization algorithms!")

        errors = []
        accuraccies_in_training_set = []
        accuraccies_in_test_set= []

        for epoch in range(number_of_epochs):
            training_X_batches = np.array_split(training_set_X, int(len(training_set_X) / self.batch_size))
            training_Y_batches = np.array_split(training_set_Y, int(len(training_set_X) / self.batch_size))

            weights_actualisation, biases_actualisation = None, None
            for Xi, Yi in zip(training_X_batches, training_Y_batches):
                y_predicted, a_vectors, z_vectors = self.process_batch(Xi)
                #print(str(weights_actualisation))
                weights_actualisation, biases_actualisation = self.backward_propagation_momentum(y_predicted, Yi, a_vectors, z_vectors, weights_actualisation, biases_actualisation)

            training_set_network_prediction = self.predict(training_set_X)
            test_set_network_prediction = self.predict(test_set_X)
            training_set_acc = np.where(training_set_Y == training_set_network_prediction,
                                        training_set_Y, 0).sum() / len(training_set_Y)
            test_set_acc = np.where(test_set_Y == test_set_network_prediction,
                                        test_set_Y, 0).sum() / len(test_set_Y)
            error_in_iteration = -(np.log(self.process_batch(test_set_X)[0]) * test_set_Y).sum(axis=1).mean()
            print("Iteration: " + str(epoch) + " Accuracy on training set: " +
                  str(training_set_acc) + " Accuracy on passed test set: " +
                  str(test_set_acc) + " Error in epoch: " + str(error_in_iteration))

            accuraccies_in_test_set.append(test_set_acc)
            accuraccies_in_training_set.append(training_set_acc)
            errors.append(error_in_iteration)

        return accuraccies_in_training_set, accuraccies_in_test_set, errors

    def train_network_nesterow_momentum(self, training_set_X, training_set_Y, test_set_X, test_set_Y, number_of_epochs):

        if(self.learning_coeffs_mode == "adadelta" or self.learning_coeffs_mode == "adagrad" or self.learning_coeffs_mode == "adam"):
            raise Exception("Momentum is not supported along with weights optimilization algorithms!")

        errors = []
        accuraccies_in_training_set = []
        accuraccies_in_test_set= []

        for epoch in range(number_of_epochs):
            training_X_batches = np.array_split(training_set_X, int(len(training_set_X) / self.batch_size))
            training_Y_batches = np.array_split(training_set_Y, int(len(training_set_X) / self.batch_size))
            weights_actualisation, biases_actualisation = None, None
            for Xi, Yi in zip(training_X_batches, training_Y_batches):
                y_predicted, a_vectors, z_vectors = self.process_batch(Xi)
                weights_actualisation, biases_actualisation = self.backward_propagation_nesterow_momentum(y_predicted, Yi, a_vectors, z_vectors, weights_actualisation, biases_actualisation)

            training_set_network_prediction = self.predict(training_set_X)
            test_set_network_prediction = self.predict(test_set_X)
            training_set_acc = np.where(training_set_Y == training_set_network_prediction,
                                        training_set_Y, 0).sum() / len(training_set_Y)
            test_set_acc = np.where(test_set_Y == test_set_network_prediction,
                                        test_set_Y, 0).sum() / len(test_set_Y)
            error_in_iteration = -(np.log(self.process_batch(test_set_X)[0]) * test_set_Y).sum(axis=1).mean()
            print("Iteration: " + str(epoch) + " Accuracy on training set: " +
                  str(training_set_acc) + " Accuracy on passed test set: " +
                  str(test_set_acc) + " Error in epoch: " + str(error_in_iteration))

            accuraccies_in_test_set.append(test_set_acc)
            accuraccies_in_training_set.append(training_set_acc)
            errors.append(error_in_iteration)

        return accuraccies_in_training_set, accuraccies_in_test_set, errors

class AdadeltaCoefficientActualiser:

    def __init__(self, weights, biases, gamma=0.9, epsilon=1e-3):
        self.epsilon = epsilon
        self.gamma = gamma
        self.last_E_g_weights = []
        self.last_E_theta_weights = []
        self.last_E_g_biases = []
        self.last_E_theta_biases = []
        for w, b in zip(weights, biases):
            self.last_E_g_weights.append(np.zeros_like(w))
            self.last_E_g_biases.append(np.zeros_like(b))
            self.last_E_theta_weights.append(np.zeros_like(w))
            self.last_E_theta_biases.append(np.zeros_like(b))

    def actualiseNetwork(self, layer_nr, weights_biases_discriminator, gradient):
        #print(str(gradient))
        if weights_biases_discriminator == "w":
            E_g = self.gamma * self.last_E_g_weights[layer_nr].mean() + (1. - self.gamma) * np.square(gradient)
        elif weights_biases_discriminator == "b":
            E_g = self.gamma * self.last_E_g_biases[layer_nr].mean() + (1. - self.gamma) * np.square(gradient)
        else:
            raise Exception("Unknown discriminator")

        delta_w = gradient / np.sqrt(E_g.mean() + self.epsilon)
        if weights_biases_discriminator == "w":
            E_w = self.gamma * self.last_E_theta_weights[layer_nr].mean() + (1 - self.gamma) * np.square(delta_w)
        elif weights_biases_discriminator == "b":
            E_w = self.gamma * self.last_E_theta_biases[layer_nr].mean() + (1 - self.gamma) * np.square(delta_w)

        RMS_w = np.sqrt(E_w)
        RMS_g = np.sqrt(E_g + self.epsilon)
        result = RMS_w / RMS_g * gradient

        if weights_biases_discriminator == "w":
            self.last_E_g_weights[layer_nr] = E_g
            self.last_E_theta_weights[layer_nr] = E_w
        elif weights_biases_discriminator == "b":
            self.last_E_g_biases[layer_nr] = E_g
            self.last_E_theta_biases[layer_nr] = E_w

        return result

class AdamActualisator:
    def __init__(self, weights, biases, beta1=0.9, beta2=0.999, epsilon=1e-2):
        self.epsilon = epsilon
        self.t = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        for w, b in zip(weights, biases):
            self.m_weights.append(np.zeros_like(w))
            self.m_biases.append(np.zeros_like(b))
            self.v_weights.append(np.zeros_like(w))
            self.v_biases.append(np.zeros_like(b))

    def actualiseNetwork(self, layer_nr, weights_biases_discriminator, gradient):
        if weights_biases_discriminator == "w":
            self.m_weights[layer_nr] = self.beta1 * self.m_weights[layer_nr] + (1 - self.beta1) * gradient
            self.v_weights[layer_nr] = self.beta2 * self.v_weights[layer_nr] + (1 - self.beta2) * (gradient ** 2)
            m_corr = self.m_weights[layer_nr] / (1 - self.beta1 ** self.t)
            v_corr = self.v_weights[layer_nr] / (1 - self.beta2 ** self.t)
        elif weights_biases_discriminator == "b":
            self.m_biases[layer_nr] = self.beta1 * self.m_biases[layer_nr] + (1 - self.beta1) * gradient
            self.v_biases[layer_nr] = self.beta2 * self.v_biases[layer_nr] + (1 - self.beta2) * (gradient ** 2)
            m_corr = self.m_biases[layer_nr] / (1 - self.beta1 ** self.t)
            v_corr = self.v_biases[layer_nr] / (1 - self.beta2 ** self.t)
        else:
            raise Exception("Unknown discriminator!")

        self.t += 1
        result = m_corr / (np.sqrt(v_corr) + self.epsilon)
        return result

#========================================================================
#               ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
#========================================================================


def softmax(X):
    #print("Another iteration")
    #print(str(X))
    nominator = np.exp(X.T)
    return (nominator / np.sum(nominator, axis=1).reshape(len(nominator), 1)).T

def softmax_(X):
    print("Another iteration")
    print(str(X))
    nominator_alt = np.exp(np.subtract(X.T,100000)) # for overflow error handling
    toRet = (nominator_alt / np.sum(nominator_alt, axis=1).reshape(len(nominator_alt), 1)).T
    print("To return")
    print(str(toRet))
    return toRet

#softmax derivative not provided here - it is hard-coded in network class instead

def relu(X):
    return np.where(X > 0, X, 0)

def relu_prime(X):
    return np.where(X > 0, 1, 0)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_prime(X):
    sig = sigmoid(X)
    return sig * (1 - sig)

def tanh(X):
    return 2 / (1 + np.exp(-2 * X)) - 1

def tanh_prime(X):
    return 1 - tanh(X)**2

#========================================================================


def flatten(X):
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])

def run(network, number_of_epochs, mode="normal"):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X = flatten(train_X[:5000])
    test_X = flatten(test_X[:5000])

    train_y = train_y[:5000].tolist()
    test_y = test_y[:5000].tolist()

    iterator = 0
    while iterator < len(train_y):
        maximal_output_index = train_y[iterator]
        x = []
        for i in range(10):
            x.append(1 if i == maximal_output_index else 0)
            train_y[iterator] = x
        iterator += 1

    iterator = 0
    while iterator < len(test_y):
        maximal_output_index = test_y[iterator]
        x = []
        for i in range(10):
            x.append(1 if i == maximal_output_index else 0)
            test_y[iterator] = x
        iterator += 1

    if mode == "normal":
        acc_train, acc_test, errors = network.train_network(train_X, train_y, test_X, test_y, number_of_epochs)
    elif mode == "momentum":
        acc_train, acc_test, errors = network.train_network_momentum(train_X, train_y, test_X, test_y, number_of_epochs)
    elif mode == "nesterow":
        acc_train, acc_test, errors = network.train_network_momentum(train_X, train_y, test_X, test_y, number_of_epochs)
    else:
        raise Error("Unknown mode!")

    return acc_test, errors, acc_train


def draw_plot(acc_test, errors):
    plt.plot(range(len(acc_test)), acc_test, label='Accuracy on test set')
    plt.plot(range(len(errors)), errors, label='Errors')
    plt.legend()
    plt.title('Accuracy of test set and errors in iterations')
    plt.xlabel('Iterations')
    plt.hlines([0, 1], 0, len(errors), linestyles='dotted', colors='black')
    plt.show()

def average_results(acc_results, errors_results):
    acc_averages = []
    iter = 0
    while iter < len(acc_results[0]):
        _sum = sum([t[iter] for t in acc_results])
        acc_averages.append(_sum / len(acc_results))
        iter += 1
    errors_averages = []
    iter = 0
    while iter < len(errors_results[0]):
        _sum = sum([t[iter] for t in errors_results])
        errors_averages.append(_sum / len(errors_results))
        iter += 1
    return acc_averages, errors_averages

def containsNan(arr):
    for x in arr:
        if math.isnan(x):
            return True
    return False

                    #par: list
                    #return: dict {network -> results} where results are average values after n iterations
def run_test(networks_to_test, experiment_number_of_executions=10, number_of_epochs = 20):
    network_results_dict = {}
    for i in networks_to_test:
        time_start = time.time()
        acc_results_for_network = []
        errors_results_for_network = []
        for k in range(experiment_number_of_executions):
            acc_test, errors, _ = run(i, number_of_epochs=number_of_epochs, mode=i.learning_mode)
            if not containsNan(acc_test) and not containsNan(errors):
                acc_results_for_network.append(acc_test)
                errors_results_for_network.append(errors)
        network_results_dict[i.name] = average_results(acc_results_for_network, errors_results_for_network)
        time_stop = time.time()
        print("\n\nNetwork=" + i.name + " TIME OF TRAINING: " + str(time_stop - time_start) + "\n\n")
    return network_results_dict


def run_experiments_task_2():
    #szybkosc uczenia i accuracy dla roznej liczby neuronow w warstwie ukrytej

    network1 = Neural_Network(relu, relu_prime, name="784 30 10", layers_sizes=(784,30,10))
    network2 = Neural_Network(relu, relu_prime, name="784 100 10", layers_sizes=(784,100,10))
    network3 = Neural_Network(relu, relu_prime, name="784 30 30 10", layers_sizes=(784,30,30,10))
    dic = run_test([network1,network2,network3], experiment_number_of_executions=10)

    for key in dic:
        print(key)
        print(str(dic[key]))

    #wplyw roznych wartosci wspolczynnika uczenia
    #
    network1 = Neural_Network(relu, relu_prime, name="0.00001", learning_coeff=0.00001)
    network2 = Neural_Network(relu, relu_prime, name="0.0001", learning_coeff=0.0001)
    network3 = Neural_Network(relu, relu_prime, name="0.001", learning_coeff=0.001)
    dic = run_test([network1,network2,network3], experiment_number_of_executions=10)

    for key in dic:
        print(key)
        print(str(dic[key]))



    #wplyw wielkowsci paczki (batcha)
    #
    network1 = Neural_Network(relu, relu_prime, name="5", batch_size=5, learning_coeff=0.0001)
    network2 = Neural_Network(relu, relu_prime, name="20", batch_size=20, learning_coeff=0.0001)
    network3 = Neural_Network(relu, relu_prime, name="30", batch_size=40, learning_coeff=0.0001)
    dic = run_test([network1,network2,network3], experiment_number_of_executions=10, number_of_epochs=20)

    for key in dic:
        print(key)
        print(str(dic[key]))

    #wplyw inicjializacji wartosci wag poczatkowych

    network1 = Neural_Network(relu, relu_prime, name="0.01", weights_sigma=0.01, biases_sigma=0.01, learning_coeff=0.0001)
    network2 = Neural_Network(relu, relu_prime, name="0.05", weights_sigma=0.05, biases_sigma=0.05, learning_coeff=0.0001)
    network3 = Neural_Network(relu, relu_prime, name="0.1", weights_sigma=0.1, biases_sigma=0.1, learning_coeff=0.0001)
    dic = run_test([network1,network2,network3], experiment_number_of_executions=3)

    for key in dic:
        print(key)
        print(str(dic[key]))

    #wplyw funkcji aktywacji (tanh lub relu)

    network1 = Neural_Network(tanh, tanh_prime, name="tanh")
    network2 = Neural_Network(relu, relu_prime, name="relu")

    dic = run_test([network1, network2], experiment_number_of_executions=3)

    for key in dic:
        print(key)
        print(str(dic[key]))


#run(network1, 4000, mode="normal")

#run(network1, 4000, mode="normal") #adadelta, adam
#network1 = Neural_Network(relu, relu_prime, name="784 50 10", layers_sizes=(784, 50, 10), batch_size=25,
#                         learning_coeff=0.02, epsilon=0.000000001, learning_coeffs_mode="adagrad")

def f_name(func):
    if func == relu:
        return "relu"
    if func == sigmoid:
        return "sigmoid"
    return "unknown"

def run_experiments_task_3():

    networks_to_test = []

    for f in ((relu, relu_prime), (sigmoid, sigmoid_prime)):
        network0 = Neural_Network(f[0], f[1], name=f_name(f[0]) + " Standard", layers_sizes=(784, 50, 10), batch_size=50,
                                  learning_coeffs_mode="none")
        network1 = Neural_Network(f[0], f[1], name=f_name(f[0]) + " Momentum", layers_sizes=(784, 50, 10), batch_size=50,
                                  learning_coeffs_mode="none", learning_mode="momentum")
        network2 = Neural_Network(f[0], f[1], name=f_name(f[0]) + " Nesterow", layers_sizes=(784, 50, 10), batch_size=50,
                                  learning_coeffs_mode="none", learning_mode="nesterow")
        network3 = Neural_Network(f[0], f[1], name=f_name(f[0]) + " Adagrad", layers_sizes=(784, 50, 10), batch_size=50,
                                  learning_coeffs_mode="adagrad", learning_coeff=0.02, epsilon=0.000000001,
                                  initialize_method="none")
        network4 = Neural_Network(f[0], f[1], name=f_name(f[0]) + " Adadelta", layers_sizes=(784, 50, 10), batch_size=50,
                                  learning_coeffs_mode="adadelta", learning_coeff=0.0001, initialize_method="none")
        network5 = Neural_Network(f[0], f[1], name=f_name(f[0]) + " Adam", layers_sizes=(784, 50, 10), batch_size=50,
                                  learning_coeffs_mode="adam", learning_coeff=0.00001, initialize_method="none")
        network6 = Neural_Network(f[0], f[1], name=f_name(f[0]) + " Adam xavier", layers_sizes=(784, 50, 10), batch_size=50,
                                  learning_coeffs_mode="adam", learning_coeff=0.00001, initialize_method="xavier")
        network7 = Neural_Network(f[0], f[1], name=f_name(f[0]) + " Adam he", layers_sizes=(784, 50, 10), batch_size=50,
                                  learning_coeffs_mode="adam", learning_coeff=0.00001, initialize_method="ha")
        networks_to_test += [network0, network1, network2, network3, network4, network5, network6, network7]

    dic = run_test(networks_to_test, experiment_number_of_executions=3,
             number_of_epochs=30)

    for key in dic:
        print(key)
        print(str(dic[key]))


run_experiments_task_3()

#run(network3, 15, mode="normal")

#prezentacja na zajeciach inicjalizacji wag
#network2 = Neural_Network(sigmoid, sigmoid_prime, name="784 50 10", layers_sizes=(784, 50, 10), batch_size=50, learning_coeffs_mode="adagrad", learning_coeff=0.02,epsilon=0.000001, initialize_method="simple")
#network3 = Neural_Network(sigmoid, sigmoid_prime, name="784 50 10", layers_sizes=(784, 50, 10), batch_size=50, learning_coeffs_mode="adagrad", learning_coeff=0.02,epsilon=0.000001, initialize_method="xavier")
#network4 = Neural_Network(sigmoid, sigmoid_prime, name="784 50 10", layers_sizes=(784, 50, 10), batch_size=50, learning_coeffs_mode="adagrad", learning_coeff=0.02,epsilon=0.000001, initialize_method="ha")

#print("no method")
#run(network2, 15, mode="normal")
#print("xavier")
#run(network3, 15, mode="normal")
#print("he")
#run(network4, 15, mode="normal")
