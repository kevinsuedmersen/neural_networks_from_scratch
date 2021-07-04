# Neural Networks from Scratch

This repository aims to derive and implement equations for training neural networks from scratch, which consist of an arbitrary number of fully connected, dense layers. First, we will attempt to derive equations for forward- as well as backward propagation in scalar form for a single training example. Then, we will extend these equations to a *matrix-based*  approach for a single training example, and finally, we will extend it to a matrix-based approach for processing `batch_size` examples at once. After implementing the necessary Python code, we will test the network's performance on the MNIST hand-written digits dataset and compare its performance with famous deep learning libraries such as TensorFlow.

# Table of contents

[TOC]

# Forward Propagation

## Forward Propagation for a Single Training Example

In this section, we will explain the forward pass, i.e. forward propagation, of a single training example. We will start by depicting the computations in component form and then continue to explain how to vectorize the forward propagation for a single training example. Finally, we will show how the network generates predictions and how it evaluates the goodness of its predictions using a specific loss function. 

Suppose we wanted to decide whether or not to go to sports today and suppose that we had three types of information, i.e. *input features*, that can aid us making that decision: The weather temperature (in degree Celsius), whether or not we slept well last night (yes or no), and whether or not we have a lot of homework to do (yes or no). To answer the question whether we should go to sports tonight, we might construct a simple neural network consisting of an input layer, one hidden layer and an output layer that might look like this: 

![neural_network_pic](resources/drawings/neural_network_pic.png)

Figure 1: Example neural network

The input layer (layer index 0) consists of 3 neurons, the hidden layer (layer index 1) consists of 2 neurons and the output layer (layer index 2) consists of 1 neuron. The input layer has one neuron per input feature $x_i$ which we will sometimes also refer to as the *activations* of the input layer, so that we may sometimes write $x_i = a_i ^0$ for $i = 1, 2, 3$. This notation allows us to represent the activations in layer $l$ in terms of the activations in layer $l-1$ for all layers $l = 0, 1,..., L$. 

The hidden layer consists of 2 neurons which are supposed to represent more complex, latent (not directly observable) features or combinations of features that the network *learns* by itself so that it can make better decisions whether or not we should go to sports today. For example, if we slept well last night and we have little homework to do, we might be in a very good *mood* today and we want to go to sports. So, some neuron in the hidden layer might be some sort of mood indicator (good or bad). 

Each of these hidden neurons has a *weighted input* $z_i^1$ and a corresponding output, i.e. activation, $a_i^1$ for $i = 1, 2$. For example, 
$$
z_1^1 = a_1^0 w_{1, 1}^1 + a_2^0 w_{1, 2}^1 + a_3^0 w_{1, 3}^1 + b_1^1
$$
$$
a_1^1 = f(z_1^1)
$$



or more generally,
$$
z_i^l = \sum_{k=1}^{n^{l-1}} \left( a_k^{l-1} w_{i, k}^{l} \right) + b_i^l
$$
$$
a_i^l = f(z_i^l),
$$



where $n^{l-1}$ represents the number of neurons in layer $l-1$, $w_{i, k}^l$ the weight that connects $a_k^{l-1}$ to $a_i^l$, $b_i^l$ represents a bias term, and where $f(\cdot)$ represents an *activation function* that is applied to the weighted input in order to produce the output, i.e. activation, of neuron $i$ in layer $l$. The weights and biases are initialized with random values[^1] and represent the parameters of the network, i.e. the parameters which the network *learns* during he training phase. The activation function $f(\cdot)$ is some non-linear transformation whose output should resemble the *firing* rate of neuron $i$ in layer $l$. A detailed discussion of activation functions will be presented later, but for now, we will just assume that our activation function is always the sigmoid function which is defined as
$$
f(z_i^l) = \frac{1}{1 + e^{-z_i^l}},
$$
which has the desirable property that $0 < f(z_i^l) < 1$, so we can say that neuron $a_i^l$ is firing if $f(z_i^l)$ is close to 1. 

[^1]: Actually, only the weights must be initialized randomly and the biases may be initialized with zeros, because only the weights suffer from the *symmetry breaking problem* (CITATION)

Then, in the output layer of our example network, we simply have one neuron that represents the probability whether or not we should go to sports, i.e. 
$$
a_1^2 = \hat{y}_i
$$
or more generally,
$$
a_i^L = \hat{y}_i
$$
where $L$ represents the final layer of the network and $\hat{y}_i$ the *probability* that we go to sports. In our example network, there is no benefit for adding the neuron index $i$, but we still left it there to show that the output layer might consists of an arbitrary number of neurons, e.g. one for each category in our classification task. Also, since $\hat{y}_i$ is a probability, we know that the activation function of the output layer must return values between $0$ and $1$. To convert the predicted probability that we will go to sports into an actual decision, we will apply a threshold as follows
$$
\text{prediction}_i =
\begin{cases}
1, & \hat{y}_i > \text{threshold}_i \\
0, & \hat{y}_i \leq \text{threshold}_i
\end{cases},
$$
where $1$ means that we will go to sports and $0$ that we won't go to sports. The threshold for category $i$ may be chosen manually and fine tuned on a validation set, but usually (CITATION), it is 0.5. If you decide to increase the threshold, your model is likely to achieve a higher precision at the expense of recall and if you decide to decrease the threshold, your model is likely to achieve a higher recall at the expense of precision[^2]. 

[^2]: Precision is the ratio of true positives divided by the sum of true positives and false positives while recall is the ratio of true positives divided by the sum of true positives and false negatives. 

Now, we want to introduce a matrix-based approach for forward propagating the input data to the output layer, because first, it will make the notation easier and second, it will make your code run faster when you actually need to implement it in Python, because vectorized operations are highly efficient and optimized. So first, we will rewrite equation (3) as 
$$
\textbf{z}^l = \textbf{W}^l \textbf{a}^{l-1} + \textbf{b}^l
$$
 or written out explicitly with all components
$$
\left[
    \matrix{
    	z_1^l \\
      	z_2^l \\
      	\vdots \\ 
      	z_{n^l}^l
    }
\right] = \left[
	\matrix{
		w_{1, 1}^l & w_{1, 2}^l & \ldots & w_{1, n^{l-1}}^l \\
		w_{2, 1}^l & w_{2, 2}^l & \ldots & w_{2, n^{l-1}}^l \\
		\vdots & \vdots & \ddots & \vdots \\
		w_{n^l, 1}^l & w_{n^l, 2}^l & \ldots & w_{n^l, n^{l-1}}^l 
	}
\right]
\left[ 
	\matrix{
		a_1^{l-^1} \\
		a_2^{l-^1} \\
		\vdots \\
		a_{n^{l-1}}^{l-^1} \\
	}
\right] + 
\left[
	\matrix{
		b_1^{l} \\
		b_2^{l} \\
		\vdots \\
		b_{n^l}^{l} \\
	}
\right]
$$
and then, equation (4) can be rewritten as
$$
\textbf{a}^l = f(\textbf{z}^l),
$$
where the activation function $f(\cdot)$ is applied element wise such as
$$
\left[ 
	\matrix{
		a_1^l \\
		a_2^l \\
		\vdots \\
		a_{n^l}^l
	}
\right] = 
\left[
	\matrix{
		f(z_1^l) \\
		f(z_2^l) \\
		\vdots \\
		f(z_{n^l}^l) \\
	}
\right].
$$
So, we just stacked the weighted inputs, the activations and the biases of each layer into column vectors $\textbf{z}^l$, $\textbf{a}^{l-1}$, and $\textbf{b}^l$. For each neuron in layer $l$, the weight matrix $\textbf{W}^l$ contains one row and for each neuron in layer $l-1$, it contains one column, meaning that the dimensions of $\textbf{W}^l$ are $n^l \times n^{l-1}$. Then finally, the activation function $f(\cdot)$ is just applied to each element of $\textbf{z}^l$ to produce the activation vector $\textbf{a}^l$. 

We can now apply equations (8) and (10) recursively all the way to the output layer $L$, until we compute the predicted probabilities of the network as follows
$$
\textbf{a}^L = \hat{\textbf{y}},
$$
or written out explicitly
$$
\left[ 
	\matrix{
		a_1^L \\
		a_2^L \\
		\vdots \\
		a_{n^L}^L
	}
\right] = 
\left[ 
	\matrix{
		\hat{y}_1 \\
		\hat{y}_2 \\
		\vdots \\
		\hat{y}_{n^L}
	}
\right],
$$
where each $\hat{y}_i$ is converted into an actual decision using (7). 

Having computed $\textbf{a}^L$, we can compute a certain *loss* which indicates how good or bad our model predicts for a single training example. For a classification problem involving $n^L$ classes, a common loss function is the binary *cross entropy* which is calculated as follows
$$
L(\textbf{y}, \hat{\textbf{y}}) = -\left( \textbf{y} log(\hat{\textbf{y}}) + (1-\textbf{y}) log(1-\hat{\textbf{y}}) \right),
$$
or written out explicitly
$$
\left[
	\matrix{
		L(y_1, \hat{y}_1) \\
		L(y_2, \hat{y}_2)\\
		\vdots \\
		L(y_{n^L}, \hat{y}_{n^L})
	}
\right] =
\left[
	\matrix{
		-\left( y_1 log(\hat{y}_1) + (1-y_1) log(1-\hat{y}_1) \right) \\
		-\left( y_2 log(\hat{y}_2) + (1-y_2) log(1-\hat{y}_2) \right) \\
		\vdots \\
		-\left( y_{n^L} log(\hat{y}_{n^L}) + (1-y_{n^L}) log(1-\hat{y}_{n^L}) \right)
	}
\right]
$$
Interpret loss function

## Forward Propagation for a Batch of Training Examples

In this section, we will explain how to vectorize the forward propagation for a batch of training examples. That is, we will show how to compute the activations of each layer and the losses for `batch_size` training examples at once. 

# Backward Propagation

The goal of the backward propagation algorithm is to iteratively adjust the weights and biases of the network such that the cost decreases. 

## Gradient Descent

## Backward Propagation in Scalar Form for a Single Training Example

## Backward Propagation in Matrix Form for a Single Training Example

## Backward Propagation in Matrix Form for a Batch of Training Examples

## Weight updates

# Loss and Activation Functions

Popular choices of activation functions in the hidden layers are the sigmoid (equation 3), ReLU (equation 4) and tanh (equation 5) functions. These functions and their corresponding derivatives are presented below
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

TODO: 

- Show equations for the activation function and their corresponding derivatives
- Show graphs of each activation function and their corresponding derivatives
- Discuss properties of the activation functions in the output layer

Ideally, $f(z)$ should be differentiable, non-linear, monotonically increasing and non-saturating. It should be differentiable, because in the back-propagation algorithm, we need to compute its derivative and it should be non-linear, because otherwise there is no benefit of introducing hidden layers. The latter follows from the fact that a chain of deeply nested linear transformations can be rewritten as merely another linear transformation (CITATION Studienbrief 3). Furthermore, it should be monotonically increasing so that as few as possible local minima are generated (GENERATED WHERE?) and it should be non-saturating to avoid the vanishing gradient problem. The vanishing gradient problem occurs when the norm of the gradient[^2] is very small which happens when $z$ is very large and $f'(z)$ is very small. Of course, the monotonically increasing and non-saturating properties could easily be achieved by setting $f(z)$ to the identity function, but then $f(z)$ would not be non-linear anymore, so there are some tradeoffs to be made and in practice, the ReLU is a good compromise between these tradeoffs. Note however, that the ReLU function is actually not defined at $z = 0$ so theoretically, it is not always differentiable. In practice however, the probability that $z$ is *exactly* $0$ is extremely small so that in code, it is often implemented that $f'(0) = 0$. 

In the output layer, the activation may also be linear and it depends whether we're doing regression or classification (multi-class or multi-label)...

[^2]: The gradient of the cost function with respect to all the weights and biases in the network

# Implementation in Code

