# Neural Networks from Scratch

This repository aims to derive and implement equations for training neural networks from scratch, which consist of an arbitrary number of fully connected, dense layers. First, we will attempt to derive equations for forward- as well as backward propagation in scalar form for a single training example. Then, we will extend these equations to a *matrix-based*  approach for a single training example, and finally, we will extend it to a matrix-based approach for processing `batch_size` examples at once. After implementing the necessary Python code, we will test the network's performance on the MNIST hand-written digits dataset and compare its performance with famous deep learning libraries such as TensorFlow.

# Table of contents

[TOC]

# Forward Propagation

TODO:

- Picture of neural network with notation explained
- Formula for activation j in layer l in scalar form
- matrix-based form (describe dimensions!)
- forward-propagation to layer L
- computation of cost function

A simple neural network consisting of an input layer, one hidden layer and an output layer might look like this: 

![neural_network_pic](resources/drawings/neural_network_pic.png)

The input layer (layer index 0) consists of 3 input features or neurons, the hidden layer (layer index 1) consists of 2 neurons and the output layer (layer index 2) consists of 1 neuron. The input layer has one neuron per input feature $x_i$ which we will sometimes also refer to as the *activations* of the input layer, so we may sometimes write $x_i = a_i ^0$ for $i = 1, 2, 3$. 

The hidden layer consists of 2 neurons. Each of these neurons has a *weighted input* $z_i^1$ and a corresponding output, i.e. activation, $a_i^1$ for $i = 1, 2$. For example, 
$$
z_1^1 = a_1^0 w_{1, 1}^1 + a_2^0 w_{1, 2}^1 + a_3^0 w_{1, 3}^1 + b_1^1 \\
a_1^1 = f(z_1^1)
$$
or more generally,
$$
z_i^l = \sum_{k=1}^{n^{l-1}} \left( a_k^{l-1} w_{i, k}^{l} \right) + b_i^l \\
a_i^l = f(z_i^l)
$$
where $n^{l-1}$ represents the number of neurons in layer $l-1$, $w_{i, k}^l$ represents the weight that connects $a_k^{l-1}$ to $a_i^l$[^1], $b_i^l$ represents a bias term, and where $f(\cdot)$ represents an *activation function* that is applied to the weighted input in order to produce the output/activation of neuron $i$ in layer $l$. 

[^1]: The weight notation may seem a little cumbersome and counter intuitive at first, but it will make more sense when we introduce the matrix notation of the feed-forward mechanism

$f(\cdot)$ should be differentiable in all layers and non-linear at least in the hidden layers. It should be differentiable, because in the back-propagation algorithm, we need to compute its derivative and it should be non-linear, because otherwise there is no benefit of introducing hidden layers. The latter follows from the fact that a chain of deeply nested linear transformations can be rewritten as merely another linear transformation (CITATION). 

Popular choices of activation functions in the hidden layers are the sigmoid (equation 3), ReLU (equation 4) and tanh (equation 5) functions. These functions and their corresponding derivatives are presented below
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


Note that the ReLU function is actually not defined at $z = 0$ so theoretically, it is not always differentiable. In practice however, the probability that $z$ is *exactly* $0$ is extremely small so that in code, it is often implemented that $f'(0) = 0$.  

In the output layer, the activation may also be linear and it depends whether we're doing regression or classification (multi-class or multi-label)...

TODO: Properties and and popular choices of activation functions in the hidden layers (see K's email)

TODO: Show activations functions and their corresponding derivatives

TODO: Properties and and popular choices of activation functions in the output layers
