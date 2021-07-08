L(y_2^1, \hat{y}_2^1)\matrix{
		z_1^{l, 1} & z_1^{l, 2} & \ldots & z_1^{l, M} \\
		z_2^{l, 1} & z_2^{l, 2} & \ldots & z_2^{l, M} \\
		\vdots & \vdots & \ddots & \vdots \\
		z_{n^l}^{l, 1} & z_{n^l}^{l, 2} & \ldots & z_{n^l}^{l, M}
	}Neural Networks from Scratch

This repository aims to derive and implement equations for training neural networks from scratch, which consist of an arbitrary number of fully connected, dense layers. First, we will attempt to derive equations for forward- as well as backward propagation in scalar form for a single training example. Then, we will extend these equations to a *matrix-based*  approach for a single training example, and finally, we will extend it to a matrix-based approach for processing `batch_size` examples at once. After implementing the necessary Python code, we will test the network's performance on the MNIST hand-written digits dataset and compare its performance with famous deep learning libraries such as TensorFlow.

# Table of contents

[TOC]

# Forward Propagation

In this section, we will start by explaining the forward pass, i.e. forward propagation, of a single training example. We will depict the computations in component form and then continue to explain how to vectorize the forward propagation for a single training example. We will show how the network generates predictions and how it evaluates the goodness of its predictions using a specific loss function. Finally, we will show a little "trick" to compute the forward propagation for `batch_size` training examples at once in a vectorized form.  

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



where $n^{l-1}$ represents the number of neurons in layer $l-1$, $w_{i, k}^l$ the weight that connects $a_k^{l-1}$ to $a_i^l$, $b_i^l$ represents a bias term, and where $f(\cdot)$ represents an *activation function* that is applied to the weighted input in order to produce the output, i.e. activation $a_i^l$, of neuron $i$ in layer $l$. The weights and biases are initialized with random values[^1] and represent the parameters of the network, i.e. the parameters which the network *learns* during the training phase. The activation function $f(\cdot)$ is some non-linear transformation whose output should resemble the *firing* rate of neuron $i$ in layer $l$. A detailed discussion of activation functions will be presented later, but for now, we will just assume that our activation function is always the sigmoid function which is defined as
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
In general, we want a loss function which has high values for bad predictions, i.e. when $\hat{y}_i$ is far away from $y_i$ and low values for good predictions, i.e. when $\hat{y}_i$ is very close to $y_i$. Let's see if component $i$ of (16) fulfills these requirements by considering the following 4 edge cases:

- Bad predictions

  - If $y_i = 1$ and $\hat{y} = 0$, then  $y_i log(\hat{y}_i) = -\infty$ and $(1-y_i) log(1 - \hat{y}_i) = 0$, so $L(y_i, \hat{y}_i) = \infty$  
  - If $y_i = 0$ and $\hat{y} = 1$, then  $y_i log(\hat{y}_i) = 0$ and $(1 - y_i) log(1 - \hat{y}_i) = -\infty$, so $L(y_i, \hat{y}_i) = \infty$  

- Good predictions

  - If $y_i = 1$ and $\hat{y} = 1$, then  $y_i log(\hat{y}_i) = 0$ and $(1 - y_i) log(1 - \hat{y}_i) = 0$, so $L(y_i, \hat{y}_i) = 0$ 

  - If $y_i = 0$ and $\hat{y} = 0$, then  $y_i log(\hat{y}_i) = 0$ and $(1 - y_i) log(1 - \hat{y}_i) = 0$, so $L(y_i, \hat{y}_i) = 0$  

in all of the 4 above cases, we get the desired result. Also, we need the loss function to be differentiable in order to compute gradients during back propagation and monotonically increasing to avoid local minima (CITATION). TODO: Read AK's email about loss function properties!

We will conclude this section by showing how to compute the complete forward propagation for `batch_size` training examples at once. It starts by defining your input data matrix as follows
$$
\textbf{X} = \textbf{A}^0,
$$
where the feature vectors of each training examples are stacked horizontally next to each other in a column-wise fashion. Assuming that we have $M$ training examples in our current batch and $n^0$ input features, equation (17) can be written out explicitly as
$$
\left[
\matrix{
	x_1^1 & x_1^2 & \ldots & x_1^M \\
	x_2^1 & x_2^2 & \ldots & x_2^M \\
	\vdots & \vdots & \ddots & \vdots \\
	x_{n^0}^1 & x_{n^0}^2 & \ldots & x_{n^0}^M
}
\right] = 
\left[ 
	\matrix{
	a_1^{0, 1} & a_1^{0, 2} & \ldots & a_1^{0, M} \\
	a_2^{0, 1} & a_2^{0, 2} & \ldots & a_2^{0, M} \\
	\vdots & \vdots & \ddots & \vdots \\
	a_{n^0}^{0, 1} & a_{n^0}^{0, 2} & \ldots & a_{n^0}^{0, M}
}
\right],
$$

where $x_i^m = a_1^{0, m}$ both refer to the value of the i-th feature of the m-th training example. 

Next, equation (9) becomes
$$
\textbf{Z}^l = \textbf{W}^l \textbf{A}^{l-1} + \textbf{B}^l,
$$
or written out explicitly
$$
\left[
	\matrix{
		z_1^{l, 1} & z_1^{l, 2} & \ldots & z_1^{l, M} \\
		z_2^{l, 1} & z_2^{l, 2} & \ldots & z_2^{l, M} \\
		\vdots & \vdots & \ddots & \vdots \\
		z_{n^l}^{l, 1} & z_{n^l}^{l, 2} & \ldots & z_{n^l}^{l, M}
	}
\right] =
\left[
	\matrix{
		w_{1, 1}^l & w_{1, 2}^l & \ldots & w_{1, n^{l-1}}^l \\
		w_{2, 1}^l & w_{2, 2}^l & \ldots & w_{2, n^{l-1}}^l \\
		\vdots & \vdots & \ddots & \vdots \\
		w_{n^l, 1}^l & w_{n^l, 2}^l & \ldots & w_{n^l, n^{l-1}}^l 
	}
\right]
\left[ 
	\matrix{
		a_1^{l-1, 1} & a_1^{l-1, 2} & \ldots & a_1^{l-1, M} \\
		a_2^{l-1, 1} & a_2^{l-1, 2} & \ldots & a_2^{l-1, M} \\
		\vdots & \vdots & \ddots & \vdots \\
		a_{n^{l-1}}^{l-1, 1} & a_{n^{l-1}}^{l-1, 2} & \ldots & a_{n^{l-1}}^{l-1, M}
	}
\right] +
\left[
	\matrix{
		b_1^{l} & b_1^{l} & \ldots & b_1^{l} \\
		b_2^{l} & b_2^{l} & \ldots & b_2^{l} \\
		\vdots & \vdots & \ddots & \vdots \\
		b_{n^l}^{l} & b_{n^l}^{l} & \ldots & b_{n^l}^{l}
	}
\right],
$$

where $\textbf{Z}^l$ simply contains $M$ columns (one for each $\textbf{z}^{l, m}$), $\textbf{A}^{l-1}$ contains $M$ columns (one for each $\textbf{a}^{l-1}$), where $\textbf{W}^l$ remains unchanged and where $\textbf{B}^l$ needs to be repeated or *broadcasted* horizontally $M$ times to make its addition conform. The dimensions of each component are as follows

- $\textbf{Z}^l \rightarrow n^l \times M$
- $\textbf{W}^l \rightarrow n^l \times n^{l-1}$
- $\textbf{A}^{l-1} \rightarrow n^{l-1} \times M$
- $\textbf{W}^l \textbf{A}^{l-1} \rightarrow n^{l} \times M$
- $\textbf{B}^l \rightarrow n^{l} \times M$

so the dimensions are conform. 

Then, the activation function is applied element-wise as usual
$$
\textbf{A}^l = f(\textbf{Z}^l),
$$
which is equal to
$$
\left[ 
	\matrix{
		a_1^{l, 1} & a_1^{l, 2} & \ldots & a_1^{l, M} \\
		a_2^{l, 1} & a_2^{l, 2} & \ldots & a_2^{l, M} \\
		\vdots & \vdots & \ddots & \vdots \\
		a_{n^{l}}^{l, 1} & a_{n^{l}}^{l, 2} & \ldots & a_{n^{l}}^{l, M}
	}
\right] =
\left[
	\matrix{
		f(z_1^{l, 1}) & f(z_1^{l, 2}) & \ldots & f(z_1^{l, M}) \\
		f(z_2^{l, 1}) & f(z_2^{l, 2}) & \ldots & f(z_2^{l, M}) \\
		\vdots & \vdots & \ddots & \vdots \\
		f(z_{n^l}^{l, 1}) & f(z_{n^l}^{l, 2}) & \ldots & f(z_{n^l}^{l, M})
	}
\right].
$$
Like before, equations (19) and (21) are applied recursively to layer $L$, until we can compute all `batch_size` losses at once, yielding the following result
$$
L(\textbf{Y}, \hat{\textbf{Y}}) = 
\left[
	\matrix{
		L(\textbf{y}^1, \hat{\textbf{y}}^1) & L(\textbf{y}^2, \hat{\textbf{y}}^2) & \ldots & L(\textbf{y}^M, \hat{\textbf{y}}^M)
	\matrix}
\right] =
\left[
	\matrix{
		L(y_1^1, \hat{y}_1^1) & L(y_1^2, \hat{y}_1^2) & \ldots & L(y_1^M, \hat{y}_1^M) \\
		L(y_2^1, \hat{y}_2^1) & L(y_2^2, \hat{y}_2^2) & \ldots & L(y_2^M, \hat{y}_2^M) \\
		\vdots & \vdots & \ddots & \vdots \\
		L(y_{n^L}^1, \hat{y}_{n^L}^2) & L(y_{n^L}^2, \hat{y}_{n^L}^2) & \ldots & L(y_{n^L}^M, \hat{y}_{n^L}^M)
	}
\right].
$$

Having computed the loss vector $L(\textbf{Y}, \hat{\textbf{Y}})$, we can now aggregate over all $M$ training examples to compute a certain *cost*. TODO: How to aggregate losses over all categories and samples

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

