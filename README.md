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

The input layer (layer index 0) consists of 3 input features, the hidden layer (layer index 1) consists of 2 neurons and the output layer (layer index 2) consists of 1 neuron. The input layer consists of three neurons, one neuron per input feature $x_i$. For more generality, we will refer to these input features also as the *activations* of the input layer, so we may sometimes write $x_i = a_i ^0$ for $i = 1, 2, 3$. 

The hidden layer consists of 2 neurons. Each of these neurons has a *weighted input* $z_i^1$ and a corresponding output, i.e. activation, $a_i^1$. For example, 
$$
z_1^1 = a_1^0 w_{1, 1}^1 + a_2^0 w_{1, 2}^1 + a_3^0 w_{1, 3}^1 + b_1^1
$$
where $w_{1, 1}^1, w_{1, 2}^1, w_{1, 3}^1$ are the weights that connect all neurons of the input layer with the first neuron in the hidden layer and where $b_1^1$ represents a bias term. 

More generally, we can re-write the equation for the weighted input as
$$
z_i^l = \sum_{k=1}^{n^{l-1}} \left( a_k^{l-1} w_{i, k}^{l} \right) + b_i
$$
where $n^{l-1}$ represents the number of neurons in layer $l-1$, and where $w_{i, k}^l$ represents the weight that connects $a_k^{l-1}$ to $a_i^l$. The weight notation may seem a little cumbersome and counter intuitive at first, but it will make more sense when we introduce the matrix notation of the feed-forward mechanism. 

Now, each weighted input $z_i^l$ is inputted into some *activation function* $f(\cdot)$ which outputs the corresponding activation value $a_i^l$, so
$$
a_i^l = f(z_i^l).
$$


The activation function $f(\cdot)$ may be linear or non-linear, but it turns out that it should be non-linear (at lease in the hidden layers), because otherwise, the neural network will only be able to learn linear functions. In other words, there would not be any benefit of adding hidden layers. 

TODO: Proof of above statement

