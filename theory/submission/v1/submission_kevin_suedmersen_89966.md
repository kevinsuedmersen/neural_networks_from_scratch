

# Neural Networks from Scratch



| Author          | Mat-Nr. | University                       | Study Program    | Course                               | Mentor              |
| --------------- | ------- | -------------------------------- | ---------------- | ------------------------------------ | ------------------- |
| Kevin Südmersen | 89966   | Hochschule Albstadt-Sigmardingen | MSc Data Science | 40200 Practical Work (Seminararbeit) | Prof. Dr. Knoblauch |



## Acknowledgments 

The knowledge used in this repository stems from a variety of sources. The below is probably an incomplete list of helpful sources, but it is definitely a good start.

- [Machine Learning course of Hochschule Albstadt-Sigmardingen (Prof. Dr. Knoblauch)](https://www.hs-albsig.de/studienangebot/masterstudiengaenge/data-science)
- [Andrew Ng's Deep Learning Courses on Coursera](https://www.coursera.org/specializations/deep-learning?utm_source=gg&utm_medium=sem&utm_campaign=17-DeepLearning-ROW&utm_content=17-DeepLearning-ROW&campaignid=6465471773&adgroupid=77415260637&device=c&keyword=coursera%20deep%20learning%20ai&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=506751438660&hide_mobile_promo&gclid=CjwKCAjw-sqKBhBjEiwAVaQ9ayqogdXIcEIxKgM1lXbJaUr4DgI5nEdHSjA9pp8Q2b3x8nFMgVo80BoCusIQAvD_BwE)
- [Micheal Nielsons online Deep Learning book](http://neuralnetworksanddeeplearning.com/)
- [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown) 

## Table of contents

[TOC]

# Introduction

This document aims to derive and implement equations for training Multi Layer Perceptrons (MLPs), i.e. Feed Forward Neural Networks consisting of a stack of dense layers, from scratch. These neural networks will consist of an arbitrary number of layers, each with an arbitrary number of neurons and arbitrary choice of activation function. At the moment, the following building blocks are supported, but may easily be extended:

- Dense layers with one of the following activation functions:
  - Sigmoid
  - Tanh
  - Linear
  - ReLU
  - Softmax
- Categorical Cross-Entropy losses
- Image data generators
- Stochastic Gradient Descent optimizers
- Following evaluation metrics:
  - Categorical Cross-Entropy
  - Accuracy (micro-averaged)
  - Precision (micro-averaged)
  - Recall (micro-averaged)

First, we will derive equations for the forward- as well as backward propagation algorithms in scalar form for a single training example. Then, we will extend these equations to a *matrix-based*  approach for a single training example, and finally, we will extend them to a matrix-based approach for a batch of training examples.

# Forward Propagation

The forward propagation algorithm propagates inputs through the layers of the network until they reach the output layer which generates the predictions. After all or a batch of input examples have been propagated through the network, the quality of these predictions are evaluated by calculating a certain cost function.    

## Forward Propagation for a Single Training Example

Suppose we wanted to decide whether or not to go to sports today and suppose that we had three types of information, i.e. *input features*, that can aid us making that decision: The weather temperature (in degree Celsius), whether or not we slept well last night (yes or no), and whether or not we have a lot of homework to do (yes or no). To answer the question whether we should go to sports tonight, we might construct a simple neural network consisting of an input layer, one hidden layer and an output layer that might look like this: 

![neural_network_pic](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/neural_network_pic.png)

Figure 1: Example neural network

The input layer (layer index 0) consists of 3 neurons, the hidden layer (layer index 1) consists of 2 neurons and the output layer (layer index 2) consists of 1 neuron. The input layer has one neuron per input feature $x_i$ which we will sometimes also refer to as the *activations* of the input layer, so that we may sometimes write $x_i = a_i ^0$ for $i = 1, 2, 3$. Later on,  this notation will allow us to represent the activations in layer $l$ in terms of the activations in layer $l-1$ for all layers $l = 0, 1,..., L$. 

The hidden layer consists of 2 neurons which are supposed to represent more complex, latent (not directly observable) features or combinations of features that the network *learns* by itself so that it can make better decisions whether or not we should go to sports today. For example, if we slept well last night and we have little homework to do, we might be in a very good *mood* today and we want to go to sports. So, some neuron in the hidden layer might be some sort of mood indicator (good or bad). 

Each of these hidden neurons has a *dendritic potential* $z_i^1$ as input and a corresponding *firing rate*, i.e. activation, $a_i^1$ as output for $i = 1, 2$. For example, 
$$
z_1^1 = a_1^0 w_{1, 1}^1 + a_2^0 w_{1, 2}^1 + a_3^0 w_{1, 3}^1 + b_1^1
$$
and 
$$
a_1^1 = \textbf{f}(z_1^1, z^1_2, z^1_3)_1
$$



or more generally,
$$
z_i^l = \sum_{k=1}^{n^{l-1}} \left( a_k^{l-1} w_{i, k}^{l} \right) + b_i^l
$$

$$
a_i^l = \textbf{f}(z^l_1, z^l_2, ..., z_i^l, ..., z^l_{n^{l-1}})_i = \textbf{f}(\textbf{z}^l)_i,
$$



where $n^{l-1}$ represents the number of neurons in layer $l-1$, $w_{i, k}^l$ the weight that connects $a_k^{l-1}$ to $a_i^l$, and where $b_i^l$ represents a bias term. Note that the weights and biases are initialized with random values and represent the parameters of the network, i.e. the parameters which the network *learns* and updates during the training phase. 

$\textbf{f}(\textbf{z}^l)_i$ represents the $i$-th output element of the *activation function* $\textbf{f}(\cdot)$ that is applied to the weighted inputs in layer $l$. Note that, in the most general case, $\textbf{f}(\cdot)$ is a function that takes a vector as input and also outputs a vector. For now however, we will assume that the activation function is simply the scalar-valued sigmoid function, which is defined as follows:
$$
f(z_i^l) = \frac{1}{1 + e^{-z_i^l}},
$$
which has the desirable property that $0 < f(z_i^l) < 1$, so we can say that neuron $a_i^l$ is firing if $f(z_i^l)$ is close to 1. 

Then, in the output layer of our example network, we simply have one neuron that represents the probability whether or not we should go to sports, i.e. 
$$
a_1^2 = \hat{y}_i
$$
or more generally,
$$
a_i^L = \hat{y}_i
$$
where $L$ represents the final layer of the network and $\hat{y}_i$ the *probability* that we go to sports. 

In our example network, there is no benefit for adding the neuron index $i$, but we still left it there to show that the output layer might consists of an arbitrary number of neurons, e.g. one for each category in our classification task. Also, since $\hat{y}_i$ is a probability, we know that the activation function of the output layer must return values between $0$ and $1$. To convert the predicted probability that we will go to sports into an actual decision, we will apply a threshold as follows
$$
\text{prediction}_i =
\begin{cases}
1, & \hat{y}_i > \text{threshold}_i \\
0, & \hat{y}_i \leq \text{threshold}_i
\end{cases},
$$
where $1$ means that we will go to sports and $0$ that we won't go to sports. 

The threshold for category $i$ may be chosen manually and fine tuned on a validation set, but for now, we will assume that $\text{threshold} = 0.5$. If you decide to increase the threshold, your model is likely to achieve a higher precision at the expense of recall and if you decide to decrease the threshold, your model is likely to achieve a higher recall at the expense of precision. Precision and recall will be defined more thoroughly later on.

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
\textbf{a}^l = \textbf{f}(\textbf{z}^l),
$$
or written out explicitly:
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
		\textbf{f}(\textbf{z}^l)_1 \\
		\textbf{f}(\textbf{z}^l)_2 \\
		\vdots \\
		\textbf{f}(\textbf{z}^l)_{n^l} \\
	}
\right].
$$
So, we just stacked the weighted inputs, the activations and the biases of each layer into column vectors $\textbf{z}^l$, $\textbf{a}^{l-1}$, and $\textbf{b}^l$. For each neuron in layer $l$, the weight matrix $\textbf{W}^l$ contains one row and for each neuron in layer $l-1$, it contains one column, meaning that the dimensions of $\textbf{W}^l$ are $n^l \times n^{l-1}$. Then finally, activation function $\textbf{f}(\cdot)$ is just applied to each element of $\textbf{z}^l$ to produce the activation vector $\textbf{a}^l$. 

We can now apply equations (9) and (11) recursively all the way to the output layer $L$, until we compute the predicted probabilities of the network as follows
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
where each $\hat{y}_i$ is converted into an actual decision using (8). 

Having computed $\textbf{a}^L$​, we can compute a certain *loss* which indicates how well or badly our model predicts for a *single* training example. For classification problems where *exactly* one of  $n^L$​ classes must be predicted (i.e. a multi-class classification problem), a commonly used loss function is the *categorical cross entropy*, which is defined as follows


$$
L(\textbf{y}, \hat{\textbf{y}}) 
= -|| \textbf{y} \ log(\hat{\textbf{y}}) ||^2 
= -\sum_{i=1}^{n^L} y_i \ log(\hat{y}_i),
$$
where $\textbf{y}$ is the ground truth vector (containing the target values) where element $y_i = 1$ and all other elements are zero if the current training example belongs to class $i$. We may also say that $\textbf{y}$ is *one-hot-encoded*. 

In general, we want a loss function which has high values for bad predictions, i.e. when $\hat{y}_i$​​ is far away from $y_i$​​, and low values for good predictions, i.e. when $\hat{y}_i$​​ is very close to $y_i$​​. Let's see if component $i$​​ of (15) fulfills these requirements by considering the following example of a multi-class classificaton problem with 2 classes:

- Bad predictions

  - If $\textbf{y} = [1, 0]^T$ and $\hat{\textbf{y}} = [0, 1]^T$, then $L(y_i, \hat{y}_i)= -(1 \times log(0) + 0 \times log(1)) = -(-\infty + 0) = \infty$
  - If $\textbf{y} = [0, 1]^T$ and $\hat{\textbf{y}} = [1, 0]^T$, then $L(y_i, \hat{y}_i)= -(0 \times log(1) + 1 \times log(0)) = -(0 -\infty) = \infty$

- Good predictions

  - If $\textbf{y} = [1, 1]^T$ and $\hat{\textbf{y}} = [1, 1]^T$, then $L(y_i, \hat{y}_i)= -(1 \times log(1) + 1 \times log(1)) = -(0 + 0) = 0$
  - If $\textbf{y} = [0, 0]^T$ and $\hat{\textbf{y}} = [0, 0]^T$, then $L(y_i, \hat{y}_i)= -(0 \times log(0) + 0 \times log(0)) = -(0 + 0) = 0$

in all of the 4 above cases, we get the desired result. 

## Forward Propagation for a Batch of  Training Examples

Assuming that we have $M$ training examples in our current batch and $n^0$ input features, imagine a 3 dimensional (3D) matrix $\textbf{X} = \textbf{A}^0$, where each element in the depth dimension belongs to a different training example: 

![X_and_A0](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/X_and_A0.png)

Figure 2

where $x^m_i = a^{l, m}_i$ represents the $i$-th activation in layer $l$ of the $m$-th training example.  

Next, equation (9) becomes
$$
\textbf{Z}^l = \textbf{W}^l \textbf{A}^{l-1} + \textbf{B}^l,
$$
or written out explicitly

![Zl](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/Zl.png)

Figure 4

where the weight matrix $\textbf{W}^l$ and the bias vector $\textbf{B}^l$ have been broad-casted $M$ times along the depth dimension in order to make the whole operation compatible. 

Note that when implementing the forward propagation for a whole batch of training examples at once in NumPy, the batch-dimension must be placed at the first axis of any 3D array (`axis=0`), i.e. every training example must be placed in a separate row.  However, in the figures above and the below figures to come, we placed each training example in the batch dimension, which is normally at `axis=2`. We chose to draw each training example in the depth dimension, because it is simply easier to draw that way. For the actual implementation in NumPy though, it's important to place each training example in a different row.  

In the implementation with NumPy, the dimensions of each component of figure 4 are as follows:

- $\textbf{Z}^l: M \times n^l \times 1$
- $\textbf{W}^l: M \times n^l \times n^{l-1}$
- $\textbf{A}^{l-1}: M \times n^{l-1} \times 1$
- $\textbf{W}^l \textbf{A}^{l-1}: M \times n^{l} \times 1$. Note here, that each of the $M$ matrix multiplications is done independently and in parallel
- $\textbf{B}^l: M \times n^{l} \times 1$

Then, like before, the activation function is applied independently to each training example

$$
\textbf{A}^l = \textbf{f}(\textbf{Z}^l),
$$
which can be written out to

![Al](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/Al.png)

Figure 5

Like before, equations (19) and (21) are applied recursively to layer $L$, until we can compute all `batch_size` losses at once, yielding the following result

![L](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/L.png)

Figure 6

where each element of the above 3D loss array represents the loss we have already defined in equation (15), i.e.
$$
L(\textbf{y}^m, \hat{\textbf{y}}^m) = -\sum_{i=1}^{n^L} y_i^m log(\hat{y}_i^m).
$$

Having computed the loss array $L(\textbf{Y}, \hat{\textbf{Y}})$, we can now aggregate over all $M$ training examples to compute a certain *cost*, which is just the average over all training examples in the current batch
$$
C = \frac{1}{M} \sum_{m=1}^M L(\textbf{y}^m, \hat{\textbf{y}}^m) 
= -\frac{1}{M} \sum_{m=1}^M \sum_{i=1}^{n^L} y_i^m log(\hat{y}_i^m),
$$
Note that the loss function represents an error over a *single* training example, while the cost function is an aggregation of the loss over $M$ training examples. When computing the cost for $M$ training examples, it makes sense to choose the average as an aggregation method, because the average cost doesn't increase linearly with the `batch_size` (like e.g. the sum). Also, the cost function may include a regularization term, which should be monotonically increasing in the number of parameters of the model, to penalize models with lots of free parameters, leading to simpler models, and hence to fight over-fitting. 

# Backward Propagation

Neural networks learn by iteratively adjusting their weights and biases such that the cost decreases, i.e. such that the predictions become more accurate. This goal is achieved by (1) computing all partial derivatives of the cost w.r.t. the weights and biases in the network (the *gradient*) and (2) by updating the weights and biases using *gradient descent*. This section will describe how to calculate, the gradient using the backpropagation algorithm which is a very cost efficient algorithm. First, the backpropagation algorithm is explained for a single training example and then extended to a whole batch of training examples. 

The backpropagation algorithm generally works as follows. For any given layer $l$​​​, the backpropagation algorithm computes an intermediate quantity, the so called *error*​​​ at layer $l$, and then computes the gradients of the weights and biases in layer $l$ using that error. Then, the error is propagated one layer backwards and the gradients are computed again. This process is repeated recursively until the gradients of the weights and biases in layer 1 (layer with index 1) are computed. 

The backpropagation algorithm is based on 4 key equations which we will derive in detail in the following sections. The four key equations are as follows:

- BP1.x: An equation for the error at the output layer, needed for initializing the backpropagation algorithm
  - When considering a single training example, we will refer to this equation as $\boldsymbol{\delta}^L$ or BP1.1
  - When considering `batch_size` training examples, we will refer to this equation as $\boldsymbol{\Delta}^L$​ or BP1.2​.
- BP2.x: A recursive equation relating the error at layer $l+1$​​​ to the error at layer $l$​​​​​​​​, needed for recursively calculating the error at each layer.
  - When considering a single training example, we will refer to this equation as $\boldsymbol{\delta}^l$ or BP2.1
  - When considering `batch_size` training examples, we will refer to this equation as $\boldsymbol{\Delta}^l$​ or BP2.2.
  - Note that in the first iteration, we must set $\boldsymbol{\delta}^l = \boldsymbol{\delta}^L$​​ or $\boldsymbol{\Delta}^l = \boldsymbol{\Delta}^L$ which we already computed in BP1.1 and BP1.2 respectively. ​​After that, we can recursively substitute the error all the way back to the input layer. 
- BP3.x: An equation relating the error at layer $l$ to:
  - The derivative of the *loss* function w.r.t the weights in layer $l$​ when considering a *single* training example, i.e. $\frac{\partial L}{\partial \textbf{W}^l}$​. We'll refer to this equation as BP3.1
  - The derivative of the *cost* function w.r.t. the weights in layer $l$​ when considering a *batch* of training examples, i.e. $\frac{\partial C}{\partial \textbf{W}^l}$​​.We'll refer to this equation as BP3.2
- BP4.x: An equation relating the error at layer $l$ to:
  - The derivative of the *loss* function w.r.t the biases in layer $l$​​​ when considering a *single* training example, i.e. $\frac{\partial L}{\partial \textbf{b}^l}$​​​. We'll refer to this equation as BP4.1
  - The derivative of the *cost* function w.r.t. the biases in layer $l$​​​​ when considering a *batch* of training examples, i.e. $\frac{\partial C}{\partial \textbf{b}^l}$​​​​​​.We'll refer to this equation as BP4.2

Most of the work will go into deriving equations BP1.1-BP4.1 and applying these equations to `batch_size` training examples at once is just a little overhead but will save a lot of time when running the actual code.  

## Why Backpropagation

You might wonder why we should bother trying to derive a complicated algorithm and not use other seemingly simpler methods for computing all partial derivatives in the network. To motivate the need for the backpropagation algorithm, assume we simply wanted to compute the partial derivative of weight $w_{j}$ as follows[^2]
$$
\frac{\partial L}{\partial w_{j}} = \text{lim}_{\epsilon \rightarrow 0} \left( \frac{L(\textbf{w} + \epsilon \textbf{e}_{j}, \textbf{b}) - L(\textbf{w}, \textbf{b})}{\epsilon} \right),
$$
where $\textbf{w}$​​ and $\textbf{b}$​​ are flattened vectors containing all weights and biases of the network, where $\epsilon$​​ is a infinitesimal scalar and where $\textbf{e}_j$​​ is the unit vector being $1$​​ at position $j$​​ and $0$​​ elsewhere. Assuming that our network has one million parameters, we would need to calculate $L(\textbf{w} + \epsilon \textbf{e}_j, \textbf{b})$​ a million times (once for each $j$​), and also, we would need to calculate $L(\textbf{w}, \textbf{b})$​ once, summing up to a total of $1,000,001$​ forward passes for just a *single* training example! As we will see in this section, the backpropagation algorithm let's us compute all partial derivatives of the network with just *one* forward- and one backward pass through the network, so the backpropagation algorithm is a very cost efficient way of computing the gradients.

[^2]: For simplicity reasons we left out the layer and column indices of the weight matrices, because this has no influence on the point we want to make. 

## Backpropagation for a Single Training Example

### BP1.1

In order to conveniently represent the error at any layer $l$, will introduce the following notation
$$
(\boldsymbol{\delta}^l)^T \coloneqq \frac{\partial L}{\partial \textbf{z}^l}.
$$
Notice that the transposition $T$ must be used, because the derivative of a scalar ($L$) w.r.t a vector ($\textbf{z}^l$) is defined as a row vector[^1].

[^1]: Notice that some authors define the derivative of a scalar w.r.t. a vector as a column vector. No matter which notation is used, the results of one notation should be equal to the transposition of the results using the other notation. 

The error at the output layer can be expressed as follows (remembering the chain rule from calculus)
$$
(\boldsymbol{\delta}^L)^T 
\coloneqq \frac{\partial L}{\partial \textbf{z}^L} 
= \frac{\partial L}{\partial \textbf{a}^L} \frac{\partial \textbf{a}^L}{\partial \textbf{z}^L}
= \nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L).
$$

The above equation shows us that the error at the output layer can be decomposed into the *gradient* $\nabla L(\textbf{a}^L)$, and the *Jacobi* matrix $\textbf{J}_{\textbf{a}^L}(\textbf{z}^L)$, from which the latter represents the derivative of a vector w.r.t. another vector. So, writing out every component of the above expression produces
$$
\nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L) = 
\left[
	\matrix{
    \frac{\partial L}{\partial a^L_1} & \frac{\partial L}{\partial a^L_2} & ... & \frac{\partial L}{\partial a^L_{n^L}}
    }
\right]
\left[
	\matrix{
    	\frac{\partial a^L_1}{\partial z^L_1} & \frac{\partial a^L_1}{\partial z^L_2} & ... & \frac{\partial a^L_1}{\partial z^L_{n^L}} \\
        \frac{\partial a^L_2}{\partial z^L_1} & \frac{\partial a^L_2}{\partial z^L_2} & ... & \frac{\partial a^L_2}{\partial z^L_{n^L}} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{\partial a^L_{n^L}}{\partial z^L_1} & \frac{\partial a^L_{n^L}}{\partial z^L_2} & ... & \frac{\partial a^L_{n^L}}{\partial z^L_{n^L}}
    }
\right],
$$
which ends up as a $1 \times n^L$ row vector. In its most general form, the above vector-matrix product represents **BP1.1**, so without making any assumptions about the loss function $L$​​ and the activation function in the output layer​​, the above equation cannot be simplified any further. In the next section, we will show how to further specify this equation by choosing a specific loss and activation function. 

#### Example

For multi-class classification problems, a common choice for the loss function is the categorical cross entropy (see equation 15) and a common choice for the activation function in the output layer is the *softmax* function, which unlike e.g. the sigmoid activation function, takes a vector as input and also outputs a vector, whose $j$​-th component is defined as follows 
$$
a^l_j = \textbf{f}([z^l_1, z^l_2, ..., z^l_j, ..., z^l_{n^l}])_j = \frac{e^{z^l_j}}{\sum_{k=1}^{n^l} e^{z^l_k}}.
$$
First, we will try to find concrete expressions for each component of $\nabla L(\textbf{a}^L)$ in equation (23). From the categorical cross entropy loss function in equation 15, we can derive that for $i = 1, 2, ..., n^L$,
$$
\frac{\partial L}{\partial a^{L}_j} = - \frac{y_j}{a^{L}_j},
$$

where we used the fact that $\hat{y}_j = a^{L}_j$​​​​​​​. 

Second, we want to find concrete expressions for each component of $\textbf{J}_{\textbf{a}^L}(\textbf{z}^L)$​​​​ in (23). When taking the derivative of the Softmax function, we need to consider two cases. The first case is represented by $\frac{\partial a^L_j}{\partial z^L_k}$​​​​, if $j=k$​​​​, i.e. $\frac{\partial a^L_j}{\partial z^L_j}$​​​​.
$$
\large
\begin{array}{l}
	\frac{\partial a^L_j}{\partial z^L_j} 
	& = \frac{e^{z^L_j} \left( \sum^{n^L}_{k=1} e^{z^l_k} \right) - e^{z^L_j} e^{z^L_j}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)^2} \\
	& = \frac{e^{z^L_j} \left( \left(\sum^{n^L}_{k=1} e^{z^L_k} \right) - e^{z^L_j} \right)}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)^2} \\
	& = \frac{e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \frac{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right) - e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \\
	& = \frac{e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \left( \frac{\sum^{n^L}_{k=1} e^{z^L_k}}{\sum^{n^L}_{k=1} e^{z^L_k}} - \frac{e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \right) \\ 
	& = \frac{e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \left( 1 - \frac{e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \right),
\end{array}
$$
 where we can now use the definition of the Softmax function (equation 28) again to simplify further to
$$
\frac{\partial a^L_j}{\partial z^L_j} = a^L_j (1 - a^L_j).
$$
The second case is represented by $\frac{\partial a^L_j}{\partial z^L_k}$​​, where $k \neq j$​​, so that
$$
\large
\begin{array}{l}
	\frac{\partial a^L_j}{\partial z^L_k}
	& = \frac{0 \times \left(\sum^{n^L}_{k=1} e^{z^L_k} \right) - e^{z^L_j} e^{z^L_k}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)^2} \\
	& = \frac{- e^{z^L_j} e^{z^L_k}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)^2} \\
	& = - \frac{e^{z^L_j}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)} \frac{e^{z^L_k}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)} \\
	& = -a^L_j \ a^L_k.
\end{array}
$$
So, summarizing, 
$$
\frac{\partial a^L_j}{\partial z^L_k} = 
	\begin{cases}
		a^L_j(1-a^L_j) & \text{if} & j=k \\
        -a^L_j \ a^L_k & \text{if} & j \neq k
	\end{cases}
$$

Using (25) and (29), we can now fill in each component of (23) as follows
$$
\nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L) = 
- \left[
	\matrix{
    	\frac{y_1}{a^L_1} & \frac{y_2}{a^L_2} & ... & \frac{y_{n^L}}{a^L_{n^L}}
    }
\right]
\left[
	\matrix{
    	a^L_1 (1 - a^L_1), & -a^L_1 \ a^L_2 & ... & -a^L_1 \ a^L_{n^L} \\
        -a^L_2 \ a^L_1, & a^L_2 (1 - a^L_2) & ... & -a^L_2 \ a^L_{n^L} \\
        \vdots & \vdots & \ddots & \vdots \\
        -a^L_{n^L} \ a^L_1, & -a^L_{n^L} \ a^L_2 & ... & a^L_{n^L} (1 - a^L_{n^L})
    }
\right],
$$

which, when multiplied out, yields
$$
\nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L) = \\
\left[
	\matrix{
		-y_1(1 - a^L_1) + y_2 a^L_1 + ... + y_{n^L} a^L_1 &
        y_1 a^L_2 - y_2(1 - a^L_2) + ... + y_{n^L} a^L_2 & 
        ... & 
        y_1 a^L_{n^L} + y_2 a^L_{n^L} + ... + (-y_{n^L})(1 - a^L_{n^L}) 
	\matrix}
\right] = \\
\left[
	\matrix{
		-y_1 + a^L_1(y_1 + y_2 + ... + y_{n^L})
		& -y_2 + a^L_2(y_1 + y_2 + ... + y_{n^L})
		& ...
		& -y_{n^L} + a^L_{n^L}(y_1 + y_2 + ... + y_{n^L}) 
	}
\right].
$$
Notice that $(y_1 + y_2 + ... + y_{n^L}) = 1$ due to the one hot encoded target vector $\textbf{y}$. So, we can simplify the above expression to
$$
\nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L) = 
- \left[
	\matrix{
		(y_1 - a^L_1) & (y_2 - a^L_2) & ... & (y_{n^L} - a^L_{n^L})
	}
\right].
$$


### BP2.1

In order to represent the error of the previous layer $(\boldsymbol{\delta}^{l-1})^T$​ in terms of the error in the current layer $(\boldsymbol{\delta}^{l})^T$, it helps to view the loss function as a nested function of weighted input vectors, i.e. $L(\textbf{z}^l(\textbf{z}^{l-1}))$ which we want to derive w.r.t. $\textbf{z}^{l-1}$. This can be done as follows
$$
(\boldsymbol{\delta}^{l-1})^T
\coloneqq \frac{\partial L}{\partial \textbf{z}^{l-1}}
= \frac{\partial L}{\partial \textbf{z}^l} \frac{\partial \textbf{z}^l}{\partial \textbf{z}^{l-1}}
= \nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}) ,
$$
which can be written out explicitly as
$$
\nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}) 
= \left[
	\matrix{
		\frac{\partial L}{\partial z^l_1} & \frac{\partial L}{\partial z^l_2} & ... & \frac{\partial L}{\partial z^l_{n^l}} 
	}
\right]
\left[
	\matrix{
		\frac{\partial z^l_1}{\partial z^{l-1}_1} & \frac{\partial z^l_1}{\partial z^{l-1}_2} & ... & \frac{\partial z^l_1}{\partial z^{l-1}_{n^{l-1}}} \\
        \frac{\partial z^l_2}{\partial z^{l-1}_1} & \frac{\partial z^l_2}{\partial z^{l-1}_2} & ... & \frac{\partial z^l_2}{\partial z^{l-1}_{n^{l-1}}} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{\partial z^l_{n^l}}{\partial z^{l-1}_1} & \frac{\partial z^l_{n^l}}{\partial z^{l-1}_2} & ... & \frac{\partial z^l_{n^l}}{\partial z^{l-1}_{n^{l-1}}}
	}
\right].
$$
In order to find an expression for every component of $\nabla L(\textbf{z}^l)$, notice that by our definition in equation (21), we have defined every element as
$$
\frac{\partial L}{\partial z^l_j} = \delta^l_j.
$$
In order to find an expression for every component of $\textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1})$, recall that $z^l_j = \sum^{n^{l-1}}_{k=1} \left( a^{l-1}_k w^l_{j, k} \right) + b^l_j$ and therefore, 
$$
\frac{\partial z^l_j}{\partial z^{l-1}_k} = \sum^{n^{l-1}}_{i=1} w^l_{j, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_k},
$$
where we need to use the total differential. To add a little more intuition why the total differential must be used here, consider the following picture and assume that we wanted to determine $\frac{\partial z^l_1}{\partial z^{l-1}_2}$. 

![total_differential_intuition](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/total_differential_intuition.png)

Figure 7 

When determining $\frac{\partial z^l_1}{\partial z^{l-1}_2}$, we want to figure out how much $z^l_1$ changes when $z^{l-1}_2$ changes (that's how derivatives are defined). In order for $z^l_1$ to change, there are 2 different paths how to achieve that:

1. Direct path: A change in $z^{l-1}_2$ leads to a change $a^{l-1}_2$, which is amplified by $w^l_{1, 2}$.
2. Indirect path: A change in $z^{l-1}_2$ might cause a change in $a^{l-1}_1 \text{ and } a^{l-1}_3$ (if e.g. the softmax activation function is used, since the activations of the softmax function alway sum up to 1), and each change in $a^{l-1}_1 \text{ and } a^{l-1}_3$ is amplified by $w^l_{1, 1}$ and $w^l_{1, 3}$ respectively.  

Using (35) and (36), we can fill in each component of (34) as follows
$$
\nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}) 
= \left[
	\matrix{
		 \delta^l_1 & \delta^l_2 & ... & \delta^l_{n^l}
	}
\right]
\left[
	\matrix{
		\sum^{n^{l-1}}_{i=1} w^l_{1, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_1}
		& \sum^{n^{l-1}}_{i=1}  w^l_{1, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_2}
		& ...
		& \sum^{n^{l-1}}_{i=1}  w^l_{1, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_{n^{l-1}}} \\
		
		\sum^{n^{l-1}}_{i=1} w^l_{2, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_1}
		& \sum^{n^{l-1}}_{i=1}  w^l_{2, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_2}
		& ...
		& \sum^{n^{l-1}}_{i=1}  w^l_{2, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_{n^{l-1}}} \\
		
		\vdots & \vdots & \ddots & \vdots \\
		
		\sum^{n^{l-1}}_{i=1} w^l_{n^l, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_1}
		& \sum^{n^{l-1}}_{i=1}  w^l_{n^l, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_2}
		& ...
		& \sum^{n^{l-1}}_{i=1}  w^l_{n^l, i} \frac{\partial a^{l-1}_i}{\partial z^{l-1}_{n^{l-1}}} \\
	}
\right],
$$
which can be decomposed into
$$
\nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}) 
= \left[
	\matrix{
		 \delta^l_1 & \delta^l_2 & ... & \delta^l_{n^l}
	}
\right]
\left[
	\matrix{
		w^l_{1, 1} & w^l_{1, 2} & ... & w^l_{1, n^{l-1}} \\
		w^l_{2, 1} & w^l_{2, 2} & ... & w^l_{2, n^{l-1}} \\
		\vdots & \vdots & \ddots & \vdots \\
		w^l_{n^l, 1} & w^l_{n^l, 2} & ... & w^l_{n^l, n^{l-1}}
	}
\right]
\left[
	\matrix{
		\frac{\partial a^{l-1}_1}{\partial z^{l-1}_1} & \frac{\partial a^{l-1}_1}{\partial z^{l-1}_2} & ... & \frac{\partial a^{l-1}_1}{\partial z^{l-1}_{n^{l-1}}} \\
		\frac{\partial a^{l-1}_2}{\partial z^{l-1}_1} & \frac{\partial a^{l-1}_2}{\partial z^{l-1}_2} & ... & \frac{\partial a^{l-1}_2}{\partial z^{l-1}_{n^{l-1}}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\frac{\partial a^{l-1}_{n^{l-1}}}{\partial z^{l-1}_1} & \frac{\partial a^{l-1}_{n^{l-1}}}{\partial z^{l-1}_2} & ... & \frac{\partial a^{l-1}_{n^{l-1}}}{\partial z^{l-1}_{n^{l-1}}}
	}
\right],
$$
or in short,
$$
(\boldsymbol{\delta}^{l-1})^T 
= \nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1})
= (\boldsymbol{\delta}^l)^T \ \textbf{W}^l \ \textbf{J}_{\textbf{a}^{l-1}}(\textbf{z}^{l-1}),
$$
or similarly, after iterating one layer forward,
$$
(\boldsymbol{\delta}^{l})^T 
= \nabla L(\textbf{z}^{l+1}) \ \textbf{J}_{\textbf{z}^{l+1}}(\textbf{z}^{l})
= (\boldsymbol{\delta}^{l+1})^T \ \textbf{W}^{l+1} \ \textbf{J}_{\textbf{a}^{l}}(\textbf{z}^{l})
$$


The above equation represents **BP2.1** in its most general form. This is a nice result, because the term $\textbf{J}_{\textbf{a}^{l-1}}(\textbf{z}^{l-1})$ already appeared in BP1.1, which shows us that for any activation function we want to use, we just need to implement its forward pass and its Jacobi matrix for the backpropagation algorithm. In other words, we regard any activation function as an interchangeable - *and completely independent* - component of our neural network. 

#### Example 

A common choice for activation functions in the hidden layers is the Sigmoid function (see equation 5), whose derivative is defined as follows
$$
\frac{\partial a^l_j}{\partial z^l_k} =
\begin{cases}
	a^l_j(1 - a^l_k) & \text{if} \ j=k \\
	0 & \text{if} \ j \neq k
\end{cases}
$$
Using the above result, we can write out (40) as follows
$$
\nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}) = \\
\left[
	\matrix{
		 \delta^l_1 & \delta^l_2 & ... & \delta^l_{n^l}
	}
\right]
\left[
	\matrix{
		w^l_{1, 1} & w^l_{1, 2} & ... & w^l_{1, n^{l-1}} \\
		w^l_{2, 1} & w^l_{2, 2} & ... & w^l_{2, n^{l-1}} \\
		\vdots & \vdots & \ddots & \vdots \\
		w^l_{n^l, 1} & w^l_{n^l, 2} & ... & w^l_{n^l, n^{l-1}}
	}
\right]
\left[
	\matrix{
		a^{l-1}_1(1 - a^{l-1}_1) & 0 & ... & 0 \\
		0 & a^{l-1}_2(1 - a^{l-1}_2) & ... & 0 \\
		\vdots & \vdots & \ddots & \vdots \\
		0 & 0 & 0 & a^{l-1}_{n^{l-1}}(1 - a^{l-1}_{n^{l-1}})
	}
\right]
,
$$

which, in this specific example, reduces to 
$$
\nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}) = \\
\left[
	\matrix{
		 \delta^l_1 & \delta^l_2 & ... & \delta^l_{n^l}
	}
\right]
\left[
	\matrix{
		w^l_{1, 1} & w^l_{1, 2} & ... & w^l_{1, n^{l-1}} \\
		w^l_{2, 1} & w^l_{2, 2} & ... & w^l_{2, n^{l-1}} \\
		\vdots & \vdots & \ddots & \vdots \\
		w^l_{n^l, 1} & w^l_{n^l, 2} & ... & w^l_{n^l, n^{l-1}}
	}
\right]
\odot
\left[
	\matrix{
		a^{l-1}_1(1 - a^{l-1}_1) & a^{l-1}_2(1 - a^{l-1}_2) & ... & a^{l-1}_{n^{l-1}}(1 - a^{l-1}_{n^{l-1}})
	}
\right]
.
$$
because $\boldsymbol{\delta}^l \ \textbf{W}^l$ yields a row vector, the Jacobian is a diagonal matrix, and a row vector multiplied with a diagonal matrix can be reformulated as the *Hadamard* product ($\odot$ operator) between the row vector and the elements on the main diagonal of the diagonal matrix. 

### BP3.1

After calculating the errors at a certain layer, we now want to relate them to the derivative of the loss w.r.t the weights in layer $l$, which we can be done as follows 

$$
\frac{\partial L}{\partial \textbf{W}^l} 
= \frac{\partial L}{\partial \textbf{z}^l} \frac{\partial \textbf{z}^l}{\partial \textbf{W}^l}
= \nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{W}^l),
$$
which can be written out explicitly as
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{W}^l) = 
\left[
	\matrix{
    	\frac{\partial L}{\partial z^l_1} & \frac{\partial L}{\partial z^l_2} & ... & \frac{\partial L}{\partial z^l_{n^l}}
        }
\right]
\left[
	\matrix{
    	\frac{\partial z^l_1}{\partial w^l_{1, 1}} & \frac{\partial z^l_1}{\partial w^l_{1, 2}} & ... & \frac{\partial z^l_1}{\partial w^l_{1, n^{l-1}}}
        & \frac{\partial z^l_1}{\partial w^l_{2, 1}} & \frac{\partial z^l_1}{\partial w^l_{2, 2}} & ... & \frac{\partial z^l_1}{\partial w^l_{2, n^{l-1}}}
        & ..., & 
        \frac{\partial z^l_1}{\partial w^l_{n^l, 1}} & \frac{\partial z^l_1}{\partial w^l_{n^l, 2}} & ... & \frac{\partial z^l_1}{\partial w^l_{n^l, n^{l-1}}} \\

		\frac{\partial z^l_2}{\partial w^l_{1, 1}} & \frac{\partial z^l_2}{\partial w^l_{1, 2}} & ... & \frac{\partial z^l_2}{\partial w^l_{1, n^{l-1}}}
        & \frac{\partial z^l_2}{\partial w^l_{2, 1}} & \frac{\partial z^l_2}{\partial w^l_{2, 2}} & ... & \frac{\partial z^l_2}{\partial w^l_{2, n^{l-1}}}
        & ... & 
        \frac{\partial z^l_2}{\partial w^l_{n^l, 1}} & \frac{\partial z^l_2}{\partial w^l_{n^l, 2}} & ... & \frac{\partial z^l_2}{\partial w^l_{n^l, n^{l-1}}} \\

		\vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\

		\frac{\partial z^l_{n^l}}{\partial w^l_{1, 1}} & \frac{\partial z^l_{n^l}}{\partial w^l_{1, 2}} & ... & \frac{\partial z^l_{n^l}}{\partial w^l_{1, n^{l-1}}}
        & \frac{\partial z^l_{n^l}}{\partial w^l_{2, 1}} & \frac{\partial z^l_{n^l}}{\partial w^l_{2, 2}} & ... & \frac{\partial z^l_{n^l}}{\partial w^l_{2, n^{l-1}}}
        & ... & 
        \frac{\partial z^l_{n^l}}{\partial w^l_{n^l, 1}} & \frac{\partial z^l_{n^l}}{\partial w^l_{n^l, 2}} & ... & \frac{\partial z^l_{n^l}}{\partial w^l_{n^l, n^{l-1}}}
    }
\right],
$$

where $J_{\textbf{z}^l}(\textbf{W}^l)$ is a $n^l \times (n^l \times n^{l-1})$ matrix, since there are $n^l$ components in $\textbf{z}^l$ and $n^l \times n^{l-1}$ components in $\textbf{W}^l$. 

Again, we will first find expressions for each component of $\nabla L(\textbf{z}^l)$ and after that, for each component of $J_{\textbf{z}^l}(\textbf{W}^l)$. The components of $\nabla L(\textbf{z}^l)$ are given by (21), and to derive each component of $J_{\textbf{z}^l}(\textbf{W}^l)$, we need to consider two cases again. 

First, consider $\frac{\partial z^l_j}{\partial w^l_{i, k}}$ if $j = i$, i.e. $\frac{\partial z^l_j}{\partial w^l_{j, k}}$.  Remember that $z^l_j = \sum^{n^{l-1}}_{k=1} w^l_{j,k} \ a^{l-1}_k + b^l_j$, so
$$
\frac{\partial z^l_j}{\partial w^l_{j,k}} = a^{l-1}_k.
$$
Next, consider $\frac{\partial z^l_j}{\partial w^l_{i, k}}$ if $j \neq i$. In that case, the weight $w^l_{i, k}$ is not connected to neuron $j$ in layer $l$, so $z^l_j$ will never change if $w^l_{i,k}$ changes and therefore,
$$
\frac{\partial z^l_j}{\partial w^l_{i,k}} = 0.
$$
Summarizing, we have that 
$$
\frac{\partial z^l_j}{\partial w^l_{i,k}} =
\begin{cases}
	a^{l-1}_k & \text{if} \ j=i \\
	0 & \text{if} \ j \neq i \\
\end{cases}
$$
Using (21) and (48), we can now fill in each value of (45) as follows
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{W}^l) = 
\left[
	\matrix{
    	\delta^l_1 & \delta^l_2 & ... & \delta^l_{n^l}
        }
\right]
\left[
	\matrix{
    	a^{l-1}_1 & a^{l-1}_2 & ... & a^{l-1}_{n^{l-1}}
        & 0 & 0 & ... & 0 
        & ... & 
        0 & 0 & ... & 0 \\

		0 & 0 & ... & 0 
        & a^{l-1}_1 & a^{l-1}_2 & ... & a^{l-1}_{n^{l-1}} 
        & ... & 
        0 & 0 & ... & 0 \\

		\vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\

		0 & 0 & ... & 0 
        & 0 & 0 & ... & 0 
        & ... 
        & a^{l-1}_1 & a^{l-1}_2 & ... & a^{l-1}_{n^{l-1}} 
    }
\right].
$$
Multiplying out the above expression yields the following $1 \times (n^l \times n^{l-1})$ row vector
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{W}^l)
= \left[
	\matrix{
		\delta^l_1 \ a^{l-1}_1 & \delta^l_1 \ a^{l-1}_2 & ... & \delta^l_1 \ a^{l-1}_{n^{l-1}} &
		\delta^l_2 \ a^{l-1}_1 & \delta^l_2 \ a^{l-1}_2 & ... & \delta^l_2 \ a^{l-1}_{n^{l-1}} &
		... &
		\delta^l_{n^l} \ a^{l-1}_1 & \delta^l_{n^l} \ a^{l-1}_2 & ... & \delta^l_{n^l} \ a^{l-1}_{n^{l-1}}
	}
\right],
$$
which we want to stack as
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{w}^l) = 
\left[
	\matrix{
		\delta^l_1 a^{l-1}_1 & \delta^l_1 a^{l-1}_2 & ... & \delta^l_1 a^{l-1}_{n^{l-1}} \\
		\delta^l_2 a^{l-1}_1 & \delta^l_2 a^{l-1}_2 & ... & \delta^l_2 a^{l-1}_{n^{l-1}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\delta^l_{n^l} a^{l-1}_1 & \delta^l_{n^l} a^{l-1}_2 & ... & \delta^l_{n^l} a^{l-1}_{n^{l-1}} \\
	}
\right],
$$
because now, we can decompose the above equation into two quantities we have already computed, i.e. 
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{W}^l) = 
\left[
	\matrix{
		\delta^l_1 \\
		\delta^l_2 \\
		\vdots \\
		\delta^l_{n^l} \\
	}
\right]
\left[
	\matrix{
		a^{l-1}_1 & a^{l-1}_2 & ... & a^{l-1}_{n^{l-1}} 
	}
\right] 
= \boldsymbol{\delta}^l \ (\textbf{a}^{l-1})^T,
$$


which represents **BP3.1**. 

### BP4.1 

Now, we want to relate the errors of each layer to the derivative of the loss w.r.t. the biases. The derivative of the loss w.r.t the biases can be expressed as follows
$$
\frac{\partial L}{\partial \textbf{b}^l} 
= \frac{\partial L}{\partial \textbf{z}^l} \frac{\partial \textbf{z}^l}{\partial \textbf{b}^l} 
= \nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{b}^l),
$$
which can be written out explicitly as
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{w}^l) =
\left[
	\matrix{
		\frac{\partial L}{\partial z^l_1} & \frac{\partial L}{\partial z^l_2} & ... & \frac{\partial L}{\partial z^l_{n^l}}
	}
\right]
\left[
	\matrix{
		\frac{\partial z^l_1}{\partial b^l_1} & \frac{\partial z^l_1}{\partial b^l_2} & ... & \frac{\partial z^l_1}{\partial b^l_{n^l}} \\
		\frac{\partial z^l_2}{\partial b^l_1} & \frac{\partial z^l_2}{\partial b^l_2} & ... & \frac{\partial z^l_2}{\partial b^l_{n^l}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\frac{\partial z^l_{n^l}}{\partial b^l_1} & \frac{\partial z^l_{n^l}}{\partial b^l_2} & ... & \frac{\partial z^l_{n^l}}{\partial b^l_{n^l}} \\
	}
\right]
$$
Again, the components of $\nabla L(\textbf{z}^l)$ are given by (21), and to derive each component of $J_{\textbf{z}^l}(\textbf{b}^l)$, we can easily see from (3) that 
$$
\frac{\partial z^l_j}{\partial b^l_k} = 
\begin{cases}
	1 & \text{if} \ j = k \\
	0 & \text{if} \ j \neq k
\end{cases}
$$
So, using (21) and (55), we can re-write equation (54) as follows
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{w}^l) =
\left[
	\matrix{
		\delta^l_1 & \delta^l_2 & ... & \delta^l_{n^l}
	}
\right]
\left[
	\matrix{
		1 & 0 & ... & 0 \\
		0 & 1 & ... & 0 \\
		\vdots & \vdots & \ddots & \vdots \\
		0 & 0 & ... & 1 \\
	}
\right],
$$
which simply means that 
$$
\frac{\partial L}{\partial \textbf{b}^l}  
= \nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{w}^l) 
= (\boldsymbol{\delta}^l)^T.
$$
The above equation represents **BP4.1**. 



## Backpropagation for a Batch of Training Examples

In the previous section, we have derived equations which can help us to compute the gradients of a *single* training example. Computing these expressions separately for each training example will take a tremendous amount of time, so in this section, we aim to extend these equations so that we can compute the gradient for `batch_size` training examples at once, harnessing already optimized and extremely efficient matrix multiplication libraries such as `NumPy`. 

### BP1.2

Recall from BP1.1 that 
$$
(\boldsymbol{\delta}^L)^T = \nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L).
$$
In order to remain conform with the notation we used when describing the forward propagation for `batch_size` training examples at once, we will first transpose both sides of the above equation to
$$
\boldsymbol{\delta}^L
= (\textbf{J}_{\textbf{a}^L}(\textbf{z}^L))^T \ (\nabla L(\textbf{a}^L))^T
= \left[
	\matrix{
    	\frac{\partial a^L_1}{\partial z^L_1} & \frac{\partial a^L_2}{\partial z^L_1} & ... & \frac{\partial a^L_{n^L}}{\partial z^L_1} \\
        \frac{\partial a^L_1}{\partial z^L_2} & \frac{\partial a^L_2}{\partial z^L_2} & ... & \frac{\partial a^L_{n^L}}{\partial z^L_2} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{\partial a^L_1}{\partial z^L_{n^L}} & \frac{\partial a^L_2}{\partial z^L_{n^L}} & ... & \frac{\partial a^L_{n^L}}{\partial z^L_{n^L}}
    }
\right]
\left[
	\matrix{
    \frac{\partial L}{\partial a^{L}_1} \\ 
    \frac{\partial L}{\partial a^{L}_2} \\ 
    \vdots \\ 
    \frac{\partial L}{\partial a^{L}_{n^L}}
    }
\right],
$$
Next, for each training example $m$, we will simply stack the Jacobian $(\textbf{J}_{\textbf{a}^{L, m}}(\textbf{z}^{L, m}))^T$ and the gradient of each training example $(\nabla L(\textbf{a}^{L, m}))^T$ for all $m = 1, 2, ..., M$ along the first axis (`axis=0`), which again, we chose to draw in the depth dimension, such that

![Delta_L](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/Delta_L.png)

Figure 8

where $\boldsymbol{\Delta}^L$ is an $M \times n^L \times 1$ dimensional array. Notice that the matrix-vector multiplications for each training example, i.e. each element in the depth dimension or where `axis=0`, is done independently and in parallel. In its most general form, the above equation represents **BP1.2**.

#### Example

Assuming we are using the categorical cross entropy cost function from equation (19) and the Softmax activation function in the output layer, we can proceed similarly as above and first transpose both sides of (32) and then stack the error of each training example in a a separate element of the depth dimension. Having done so, we will end up with the following expression

![Delta_L_example](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/Delta_L_example.png)

Figure 9

which is easily computed, because we are given $\textbf{Y}$ (since we are talking about a supervised learning problem here) and we already computed $\textbf{A}^L$ during the forward propagation. 

### BP2.2

Recall from BP2.1 that 
$$
(\boldsymbol{\delta}^{l-1})^T = (\boldsymbol{\delta}^l)^T \ \textbf{W}^l \ \textbf{J}_{\textbf{a}^{l-1}}(\textbf{z}^{l-1}).
$$
Again, we will first transpose both sides of the above equation giving us
$$
\boldsymbol{\delta}^{l-1} 
= (\textbf{J}_{\textbf{a}^{l-1}}(\textbf{z}^{l-1}))^T \ (\textbf{W}^l)^T \ \boldsymbol{\delta}^l.
$$

As before, we want to stack each error $\boldsymbol{\delta}^{l, m}$ and each Jacobian $(\textbf{J}_{\textbf{a}^{l-1, m}}(\textbf{z}^{l-1, m}))^T$ along `axis and broadcast each weight matrix $ (\textbf{W}^l)^T$ for all $m = 1, 2, ..., M$ training examples in each element of the depth dimension, such that

![Delta_l_1.png](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/Delta_l_1.png)

Figure 10

where $\Delta^{l-1}$ is a $M \times n^{l-1} \times 1$ dimensional array. Again, the matrix-matrix-vector multiplication for each training example is done independently. In its most general form, Figure 10 represents **BP2.2**. 

#### Example

Using the sigmoid activation function in layer $l-1$, figure 10 can be specified as follows

![Delta_l_1_example](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/Delta_l_1_example.png)

Figure 11

### BP3.2

Remember that the real quantities of interest during backpropagation are the gradients of the *cost* function w.r.t. the weights and biases, because we need those to adjust the weights and biases into the direction so that the overall cost decreases. Also, recall that the cost is just the averaged loss over $M$ training examples, i.e. 
$$
\frac{\partial C}{\partial \textbf{W}^l} = \frac{1}{M} \sum^M_{m=1} \frac{\partial L^m}{\partial \textbf{W}^l},
$$
where $L^m$ is the loss associated with the $m$-th training example. 

From BP3.1, we know that 
$$
\frac{\partial L}{\partial \textbf{W}^l} = 
\left[
	\matrix{
		\delta^l_1 \\
		\delta^l_2 \\
		\vdots \\
		\delta^l_{n^l}
	}
\right]
\left[
	\matrix{
		a^{l-1}_1 & a^{l-1}_2 & ... & a^{l-1}_{n^{l-1}}
	}
\right],
$$
so, using that, we can rewrite (62) as follows
$$
\frac{\partial C}{\partial \textbf{W}^l} = 
\frac{1}{M} \sum^M_{m=1} 
\left[
	\matrix{
		\delta^{l, m}_1 \\
		\delta^{l, m}_2 \\
		\vdots \\
		\delta^{l, m}_{n^l}
	}
\right]
\left[
	\matrix{
		a^{l-1, m}_1 & a^{l-1, m}_2 & ... & a^{l-1, m}_{n^{l-1}}
	}
\right].
$$
Using the same notation as before, we can represent the above equation such that each training example belongs to a different element of the depth dimension

![dC_dW_l](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/dC_dW_l_1.png)

Figure 12

where $np.mean$ refers to the `mean` function of `NumPy` . Figure 12 can be multiplied out as

 ![dC_dW_l_2](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/dC_dW_l_2.png)

Figure 13

Note that the average is taken along `axis=0`, i.e. we take the average across all training examples of each element of the matrix resulting from $\boldsymbol{\delta}^{l} (\textbf{a}^{l-1})^T$. Figure 13 represents **BP3.2**, where $\frac{\partial C}{\partial \textbf{W}^l}$ is an $M \times n^l \times n^{l-1}$ dimensional array. 

### BP4.2

Finally, the other real quantity of interest is the gradient of the cost function w.r.t. the biases. Again, using the fact that the cost is an average over $M$​ training examples, we can deduce that
$$
\frac{\partial C}{\partial \textbf{b}^l}
= \frac{1}{M} \sum^M_{m=1} \frac{\partial L^m}{\partial \textbf{b}^{l}}.
$$
Remember from BP4.1 that 
$$
\frac{\partial L}{\partial \textbf{b}^{l}}
= \left[
	\matrix{
		\delta^l_1 & \delta^l_2 & ... & \delta^l_{n^l}
	}
\right].
$$
First of all, we will transpose both sides of the above equation in order to remain conform with the notations used from BP1.2 to BP3.2, i.e.
$$
\left( \frac{\partial L}{\partial \textbf{b}^{l}} \right)^T= 
\left[
	\matrix{
		\delta^l_1 \\ 
		\delta^l_2 \\
        \vdots \\
        \delta^l_{n^l}
	}
\right].
$$
From here on out, it is really straight forward. Transpose both sides of (65) and use (67), yielding
$$
\left( \frac{\partial C}{\partial \textbf{b}^l} \right)^T = 
\frac{1}{M} \sum^M_{m=1}
\left[
	\matrix{
		\delta^{l, m}_1 \\ 
		\delta^{l, m}_2 \\
        \vdots \\
        \delta^{l, m}_{n^l}
	}
\right].
$$
Representing (68) such that each training example refers to a separate element of the depth dimension, we will get

![dC_db_l](C:/Users/ksu/dev/UNI/neural_networks_from_scratch/theory/dC_db_l.png)

Figure 14

where again, we take the average across all training examples of each element of $\boldsymbol{\delta}^l$. Figure 14 represents **BP4.2**, where $\left( \frac{\partial C}{\partial \textbf{b}^l} \right)^T$ is an $M \times n^l \times 1$ dimensional array. 

# Gradient Descent

In the previous section, we described how to compute the gradients, which mathematically speaking, point into the *direction* of the steepest ascent of the cost function. In this section, we will describe how to use the gradients in order to *update* the weights and biases such that the cost decreases. 

We will describe a very simple way of updating the weights and biases which is called *Stochastic Gradient Descent* (SGD). Assuming that we have calculated BP3.2 and BP4.2 for all layers, we can perform the weight updates as
$$
\textbf{W}^{l}_{s} = \textbf{W}^{l}_{s-1} - \lambda \left( \frac{\partial C}{\partial \textbf{W}^l} \right)_{s-1},
$$
and similarly, the bias updates as
$$
\textbf{b}^{l}_s = \textbf{b}^{l}_{s-1} - \lambda \left( \frac{\partial C}{\partial \textbf{b}^l} \right)^T_{s-1},
$$
for update steps $i = 1, 2, ..., S$. Notice that $\textbf{w}^{l}_{s=0}$ and $\textbf{b}^{l}_{s=0}$ are initialized randomly, $S$ represents the number of update steps and $\lambda$ represents the *learning rate* controlling the step size toward the local (and hopefully global) minimum of the cost function. 

Assuming that we divided our dataset into batches with at most $M$ training examples each, we will end up with $S$ batches, where $S$ is computed as 
$$
S = \text{round\_up}(N/M),
$$
where $\text{round\_up}$​ is a function that always rounds a floating point number *up* to the nearest integer and where $N$ represents the number of all training examples in total. Notice that $S$ always needs to be rounded up in order to make sure that during one *epoch*[^3], all training examples have been forward- and backward propagated through the network. If we rounded down, some training examples might be skipped. 

[^3]: During one epoch, all training examples have been forward- and backward propagated through the network. Usually, neural networks will need many (50-100) of such epochs to accurately predict the target values. Notice, that during each epoch, $S$​ gradient descent update steps are performed.  

Note that, before updating the weights and biases of each layer, we must have calculated the errors of *all* layers beforehand. Calculating the gradients and updating the weights and biases for each layer simultaneously will yield incorrect gradients, because in BP2.1 and BP2.2, we can see that the error of layer $l$ is also a function of the weights in layer $l+1$. So, if you were to calculate the error of layer $l+1$ and then immediately update the weights in layer $l+1$, you will get a wrong error for layer $l$ in the next iteration. To get the correct errors of each layer, you need to keep the weights fixed. Only after having calculated the errors of each layer, you may update the weights and biases.  

# Loss functions

This section will show how all relevant loss functions and their gradients $\nabla L (\textbf{a}^L) = \frac{\partial L}{\partial \textbf{a}^L}$, which are needed as an interim quantity to initialize the error at the output layer (see BP1.1 and BP1.2). 

## Categorical Crossentropy

Recall from (15) that
$$
L = - \sum^{n^L}_{i=1} y_i \ log(a^{L}_i),
$$
where we used that $\hat{\textbf{y}}^m = \textbf{a}^{L, m}$. Its gradient is defined as
$$
\frac{\partial L}{\partial \textbf{a}^{L}}
= - \frac{\textbf{y}}{\textbf{a}^L}
= - \left[
	\matrix{
		\frac{y_1}{a^L_1} & \frac{y_2}{a^L_2} & ... & \frac{y_{n^L}}{a^L_{n^L}}
	}
\right]
$$


## Sum of Squared Errors

The Sum of Squared Errors loss is defined as
$$
L = \frac{1}{2} \sum^{n^L}_{i=1} \left( y_i - a^L_i \right)^2
$$
and its gradient is
$$
\frac{\partial L}{\partial \textbf{a}^{L}}
= - (\textbf{y} - \textbf{a}^L)
= - \left[
	\matrix{
		(y_1 - a^L_1) & (y_2 - a^L_2) & ... & (y_{n^L} - a^L_{n^L})
	}
\right]
$$


# Activation Functions

This section will show all relevant activations functions and their corresponding Jacobians $\textbf{J}_{\textbf{a}^l}(\textbf{z}^l)$ which are needed as an interim quantity when initializing the error at the output layer (see BP1.1 and BP1.2) as well as an interim quantity when backpropagating the error from layer to layer (see BP2.1 and BP2.2). Notice that for all activation functions shown here, the Jacobians are always symmetric matrices, so in the actual Python implementation of the activation functions, we may use that $(\textbf{J}_{\textbf{a}^l}(\textbf{z}^l))^T = \textbf{J}_{\textbf{a}^l}(\textbf{z}^l)$. 

## Sigmoid

Recall from (5) that
$$
a^l_i = f(z^l_i) = \frac{1}{1 + e^{-z^l_i}}.
$$

 From (45) and 46), we can infer that the Jacobian of the Sigmoid function is
$$
\textbf{J}_{\textbf{a}^{l}}(\textbf{z}^l) 
= \left[
	\matrix{
		a^{l}_1(1 - a^{l}_1) & 0 & ... & 0 \\
		0 & a^{l}_2(1 - a^{l}_2) & ... & 0 \\
		\vdots & \vdots & \ddots & \vdots \\
		0 & 0 & 0 & a^{l}_{n^{l}}(1 - a^{l}_{n^{l}})
	}
\right].
$$

### Numerical Stability Amendment 

Imagine the input to the sigmoid function $z^l_i$ is largely negative. In that case, the exponent of $e$ would be extremely large and in that case, $e^{z^l_i}$ would converge to infinity very quickly. If you were to implement the above version of the sigmoid function Python, you will most likely encounter a situation where at some point, $e^{z^l_i} = \infty$ and if $\infty$ is used in subsequent computations, you will get some error. To avoid this error, we will use the following, equivalent version of the sigmoid function for any negative $z^l_i$.
$$
a^l_i = f(z^l_i) = \frac{e^{z^l_i}}{1 + e^{z^l_i}}
$$
To see how the above amendment equals the original sigmoid function, see the following proof.
$$
\frac{e^{z^l_i}}{1 + e^{z^l_i}} = \\
\frac{e^{z^l_i}}{1 + e^{z^l_i}} \frac{e^{-z^l_i}}{e^{-z^l_i}} = \\
\frac{1}{e^{-z^l_i} + e^{z^l_i} e^{-z^l_i}} = \\
\frac{1}{1 + e^{-z^l_i}}
$$
For large positive $z^l_i$, $e^{-z^l_i}$ will convert to $0$, in which case we can use the original version of the sigmoid function in (78). 

## ReLU

The Rectified Linear Unit (ReLU) is defined as
$$
a^l_i = f(z^l_i) = max(0, z^l_i).
$$
Its derivative is actually not defined for $z=0$​, but in practice, the probability that $z=0.000000000...$​ exactly is infinitesimal. So, we will define the derivative of the ReLU function as 
$$
\frac{\partial a^l_i}{\partial z^l_i}
= f'(z^l_i) 
= \begin{cases}
1 & \text{if} & z^l_i > 0 \\
0 & \text{if} & z^l_i \leq 0
\end{cases}.
$$
Since like the Sigmoid function, ReLU function also just depends on a single scalar, we know that
$$
\frac{\partial a^l_i}{\partial z^l_j} =
\begin{cases}
	f'(z^l_i) & \text{if} \ i=j \\
	0 & \text{if} \ i \neq k
\end{cases}
$$
where $f'(z^l_i)$ was already defined in (81). Using (82), we can construct Jacobian of the ReLU as follows 
$$
\textbf{J}_{\textbf{a}^{l}}(\textbf{z}^l) 
= \left[
	\matrix{
		f'(z^l_1) & 0 & ... & 0 \\
		0 & f'(z^l_2) & ... & 0 \\
		\vdots & \vdots & \ddots & \vdots \\
		0 & 0 & 0 & f'(z^l_{n^l})
	}
\right].
$$


## tanh

The tanh function is defined as follows
$$
a^l_i = f(z^L_i) = \frac{e^{z^l_i} - e^{-{z^l_i}}}{e^{z^l_i} + e^{-{z^l_i}}}
$$
and one can show that its derivative is
$$
\frac{\partial a^l_i}{\partial z^l_j} = 
\begin{cases}
	1 - (a^l_i)^2 & \text{if} \ i = j \\
	0 & \text{if} \ i \neq j
\end{cases}. 
$$
Hence, its Jacobian is
$$
\textbf{J}_{\textbf{a}^{l}}(\textbf{z}^l) 
= \left[
	\matrix{
		1 - (a^l_1)^2 & 0 & ... & 0 \\
		0 & 1 - (a^l_2)^2 & ... & 0 \\
		\vdots & \vdots & \ddots & \vdots \\
		0 & 0 & 0 & 1 - (a^l_{n^l})^2
	}
\right].
$$

### Numerical Stability Amendment

Unlike the sigmoid function, the tanh function will suffer from numerical instability if both $z^l_i \rightarrow \infty$ and $z^l_i \rightarrow -\infty$, because once the exponent is positive and once it is negative in (84). To avoid numerical instability for large positive $z^l_i$, we will introduce the following equivalent version:
$$
a^l_i = f({z^l_i}) = \frac{1 - e^{-2{z^l_i}}}{1 + e^{-2{z^l_i}}}
$$
Proof:
$$
\frac{1 - e^{-2{z^l_i}}}{1 + e^{-2{z^l_i}}} = \\
\frac{1 - e^{-2{z^l_i}}}{1 + e^{-2{z^l_i}}} \frac{e^{z^l_i}}{e^{z^l_i}} = \\
\frac{e^{z^l_i} - e^{z^l_i} \ e^{-2{z^l_i}}}{e^{z^l_i} + e^{z^l_i} \ e^{-2{z^l_i}}} = \\
\frac{e^{z^l_i} - e^{-z^l_i}}{e^{z^l_i} + e^{-z^l_i}},
$$
Now, for large $z^l_i$, $e^{-2 z^l_i}$ will approach $0$. 

For large negative inputs, we will implement the following, equivalent version:
$$
a^l_i = f({z^l_i}) = \frac{e^{2{z^l_i}} - 1}{e^{2{z^l_i}} + 1}
$$
Proof:
$$
\frac{e^{2{z^l_i}} - 1}{e^{2{z^l_i}} + 1} = \\
\frac{e^{2{z^l_i}} - 1}{e^{2{z^l_i}} + 1} \frac{e^{-z^l_i}}{e^{-z^l_i}} = \\
\frac{e^{2{z^l_i}} \ e^{-{z^l_i}} - e^{-{z^l_i}}}{e^{2{z^l_i}} \ e^{-{z^l_i}} + e^{-{z^l_i}}} = \\
\frac{e^{z^l_i} - e^{-z^l_i}}{e^{z^l_i} + e^{-z^l_i}},
$$
Now, for large negative $z^l_i$, $e^{2 z^l_i}$ will also approach $0$.  

## Softmax

Unlike all other activation functions discussed so far, the Softmax function is a vector valued function which receives a vector of length $n$ as input and also outputs a vector of length $n$, whose elements sum up to $1$. Recall from (24) that the Softmax function is defined as
$$
a^l_i = \textbf{f}([z^l_1, z^l_2, ..., z^l_i, ..., z^l_{n^l}])_i = \frac{e^{z^l_i}}{\sum_{k=1}^{n^l} e^{z^l_k}}
$$
and recall from (30), that its Jacobian is
$$
\textbf{J}_{\textbf{a}^l}(\textbf{z}^l) = 
\left[
	\matrix{
    	a^l_1 (1 - a^l_1), & -a^l_1 \ a^l_2 & ... & -a^l_1 \ a^l_{n^l} \\
        -a^l_2 \ a^l_1, & a^l_2 (1 - a^l_2) & ... & -a^l_2 \ a^l_{n^l} \\
        \vdots & \vdots & \ddots & \vdots \\
        -a^l_{n^l} \ a^l_1, & -a^l_{n^l} \ a^l_2 & ... & a^l_{n^l} (1 - a^l_{n^l})
    }
\right].
$$

### Numerical Stability Amendment

Because the exponent $e^{z^l_i}$ is positive, we might run into numerical overflow problems if $z^l_i \rightarrow \infty$. So, we should try to reduce $e^{z^l_i}$ somehow without changing the result of (91). The following implementation will do just that:
$$
a^l_i = \textbf{f}([z^l_1, z^l_2, ..., z^l_i, ..., z^l_{n^l}])_i = \\
\textbf{f}(\textbf{z}^l)_i = \frac{e^{z^l_i - \text{max}(\textbf{z}^l)}}{\sum_j^{n^l} e^{z^l_j - \text{max}(\textbf{z}^l)}}
$$
Proof:
$$
\frac{e^{z^l_i - \text{max}(\textbf{z}^l)}}{\sum_j^{n^l} e^{z^l_j - \text{max}(\textbf{z}^l)}} = \\
\frac{e^{z^l_i} \ e^{-\text{max}(\textbf{z}^l)}}{\sum_j^{n^l} e^{z^l_j} \ e^{-\text{max}(\textbf{z}^l)}} = \\
\frac{e^{z^l_i}}{\sum_{j=1}^{n^l} e^{z^l_j}} \frac{e^{-\text{max}(\textbf{z}^l)}}{e^{-\text{max}(\textbf{z}^l)}} = \\
\frac{e^{z^l_i}}{\sum_{j=1}^{n^l} e^{z^l_j}},
$$


where $\text{max}(\textbf{z}^l)$ represents the maximum of $\textbf{z}^l$, which could actually be replaced by any other constant. However, $\text{max}(\textbf{z}^l)$ seems like a good choice because by definition, it is a large value and hence keeps the exponent small. Notice that we can take $e^{-\text{max}(\textbf{z}^l)}$ out of the summation, because it will reduce to just some constant. 

# Implementation

The actual Python implementation can be found in the `src/lib` directory which contains the following packages:

- `activation_functions` : Contains separate modules (i.e. python files) for each activation function which all have the same signature so that they can be used interchangeably. For each activation function, the forward and backward pass are implemented as described in the [Activation Functions](#Activation Functions) section
- `data_generators` : Contains data generators yielding batches of training data for supervised learning problems. Each type of data generator must implement the `train`, `val`, and `test` method so that they can be used interchangeably. Currently, an image data generator has been implemented. 
- `layers` : Contains different layer types responsible for forward- and backpropagation as well as gradient computations. Each layer type must implement the `init_parameters`, `forward_propagate`, `backward_propagate`, `compute_weight_gradients`, and `compute_bias_gradients` methods. 
- `losses` : Implements the loss as well as cost computation and initializes backpropagation. Each loss must implement `compute_losses`, `compute_cost`, and `init_error` methods. We chose to initialize the backpropagation in this class, because initializing backpropagation is dependent on the choice of loss/cost function.
- `metrics` : Implements different metrics for evaluating performance during or after training. Each metric must implement the `update_state`, `result`, and `reset_state` methods. 
- `optimizers` : Implements the weights and biases updates. Each optimizer must implement the `update_parameters` method and at the moment, Stochastic Gradient Descent is supported. 
- `models` : Implements different types of models in which all of the above building blocks come together and are executed in a specific way. Each type of model must implement the `train_step`, `fit`, `predict`, `val_step`, and `evaluate` methods. Currently sequential models are supported which consist of a stack of layers executed sequentially. 

# Tests

A lot of tests can be found in the `tests` folder which generally contains one module per package described above. The most interesting tests are contained in `tests/test_model.py` which takes a simple example neural network and tests that the forward pass produces the expected dendritic potentials, activations, losses and cost and that the backward pass produces the expected gradients. The expected results were computed manually and the actual results were computed using the above described algorithms for forward- and backward propagation. 

The architecture the example network is as follows:

- Input layer: 3 neurons
- Hidden layer: 2 neurons with sigmoid activation function
- Output layer: 2 neurons with softmax activation function and Categorical Cross -ntropy loss

As the forward pass computations are relatively trivial and have already been explained in the [Forward Propagation](#Forward Propagation) section, we will now show how the gradients of each layer are computed manually. 

Since our example network has two layers with trainable parameters, i.e. the hidden and output layer, what we are most interested in are $\frac{\partial L}{\partial \textbf{W}^1}$ and $\frac{\partial L}{\partial \textbf{b}^1}$ as well as $\frac{\partial L}{\partial \textbf{W}^2} $ and $\frac{\partial L}{\partial \textbf{b}^2}$. Each of these terms can be expanded using the chain rule. We will start with the gradients in the hidden layer:
$$
\frac{\partial L}{\partial \textbf{W}^2} = 
\frac{\partial L}{\partial \textbf{a}^2} 
\frac{\partial \textbf{a}^2}{\partial \textbf{z}^2} 
\frac{\partial \textbf{z}^2}{\partial \textbf{W}^2},
$$
where 
$$
\frac{\partial L}{\partial \textbf{a}^2} = 
\left[
	\matrix{
		\frac{\partial L}{\partial a^2_1} & \frac{\partial L}{\partial a^2_2}
	}
\right] = 
-\left[
	\matrix{
		\frac{y_1}{a^2_1} & \frac{y_2}{a^2_2}
	}
\right],
$$

(gradient of the categorical cross entropy loss)
$$
\frac{\partial \textbf{a}^2}{\partial \textbf{z}^2} =
\left[
	\matrix{
		\frac{\partial a^2_1}{\partial z^2_1} & \frac{\partial a^2_1}{\partial z^2_2} \\
		\frac{\partial a^2_2}{\partial z^2_1} & \frac{\partial a^2_2}{\partial z^2_2}
	}
\right] = 
\left[
	\matrix{
		a^2_1(1 - a^2_1) & -a^2_1 a^2_2 \\
		-a^2_2 a^2_1 & a^2_2(1 - a^2_2)
	}
\right],
$$

(Jacobian of the softmax activation function)
$$
\frac{\partial \textbf{z}^2}{\partial \textbf{W}^2} =
\left[
	\matrix{
		\frac{\partial z^2_1}{\partial w^2_{1,1}} & \frac{\partial z^2_1}{\partial w^2_{1,2}} & \frac{\partial z^2_1}{\partial w^2_{2,1}} & \frac{\partial z^2_1}{\partial w^2_{2,2}} \\
		\frac{\partial z^2_2}{\partial w^2_{1,1}} & \frac{\partial z^2_2}{\partial w^2_{1,2}} & \frac{\partial z^2_2}{\partial w^2_{2,1}} & \frac{\partial z^2_2}{\partial w^2_{2,2}}
	}
\right] = 
\left[
	\matrix{
		a^1_1 & a^1_2 & 0 & 0 \\
		0 & 0 & a^1_1 & a^1_2
	}
\right],
$$

(see equation (3))

which, after forward propagation, can be filled in with exact values. As a sanity check, note that (95) will yield a $1 \times 4$ row vector (or stacked as a $2 \times 2$ matrix if you will), i.e. it has 4 elements, just as many elements as $\textbf{W}^2$ should have. 

Having calculated $\frac{\partial L}{\partial \textbf{W}^2}$, calculating $\frac{\partial L}{\partial \textbf{b}^2}$ is straight forward, because we already know some intermediate quantities:
$$
\frac{\partial L}{\partial \textbf{b}^1} = 
\frac{\partial L}{\partial \textbf{a}^2} 
\frac{\partial \textbf{a}^2}{\partial \textbf{z}^2} 
\frac{\partial \textbf{z}^2}{\partial \textbf{b}^2}
,
$$
where the new quantity
$$
\frac{\partial \textbf{z}^2}{\partial \textbf{b}^2} = 
\left[
	\matrix{
		1 & 0 \\
		0 & 1
	}
\right]
$$

(see equation (3)), which will yield a $1 \times 2$ row vector, as expected. 

Now, let's move to the weights in the first layer: 
$$
\frac{\partial L}{\partial \textbf{W}^1} = 
\frac{\partial L}{\partial \textbf{a}^2} 
\frac{\partial \textbf{a}^2}{\partial \textbf{z}^2}
\frac{\partial \textbf{z}^2}{\partial \textbf{a}^1}
\frac{\partial \textbf{a}^1}{\partial \textbf{z}^1}
\frac{\partial \textbf{z}^1}{\partial \textbf{W}^1},
$$

where the new quantities are
$$
\frac{\partial \textbf{z}^2}{\partial \textbf{a}^1} = \left[
	\matrix{
		\frac{\partial z^2_1}{\partial a^1_1} & \frac{\partial z^2_1}{\partial a^1_2} \\
		\frac{\partial z^2_2}{\partial a^1_1} & \frac{\partial z^2_2}{\partial a^1_2}
	}
\right] = 
\left[
	\matrix{
		w^2_{1,1} & w^2_{1,2} \\
		w^2_{2,1} & w^2_{2,2}
	}
\right],
$$
(see equation (3))
$$
\frac{\partial \textbf{a}^1}{\partial \textbf{z}^1} = 
\left[
	\matrix{
		\frac{\partial a^1_1}{\partial z^1_1} & \frac{\partial a^1_1}{\partial z^1_2} \\
		\frac{\partial a^1_2}{\partial z^1_1} & \frac{\partial a^1_2}{\partial z^1_2}
	}
\right] = 
\left[
	\matrix{
		a^1_1(1 - a^1_1) & 0 \\
		0 & a^1_2(1 - a^1_2)
	}
\right],
$$
(Jacobian of the sigmoid function)
$$
\frac{\partial \textbf{z}^1}{\partial \textbf{W}^1} = 
\left[
	\matrix{
		\frac{\partial z^1_1}{\partial w^1_{1,1}} & \frac{\partial z^1_1}{\partial w^1_{1,2}} & \frac{\partial z^1_1}{\partial w^1_{1,3}} & \frac{\partial z^1_1}{\partial w^1_{2,1}} & \frac{\partial z^1_1}{\partial w^1_{2,2}} & \frac{\partial z^1_1}{\partial w^1_{2,3}} \\
		\frac{\partial z^1_2}{\partial w^1_{1,1}} & \frac{\partial z^1_2}{\partial w^1_{1,2}} & \frac{\partial z^1_2}{\partial w^1_{1,3}} & \frac{\partial z^1_2}{\partial w^1_{2,1}} & \frac{\partial z^1_2}{\partial w^1_{2,2}} & \frac{\partial z^1_2}{\partial w^1_{2,3}}
	}
\right] = 
\left[
	\matrix{
		a^0_1 & a^0_2 & a^0_3 & 0 & 0 & 0 \\
		0 & 0 & 0 & a^0_1 & a^0_2 & a^0_3
	}
\right]
$$
(see equation (3)).

Finally, the computation of the bias gradients of layer 1 is very similar to the one of the weight gradients of layer 1, i.e.
$$
\frac{\partial L}{\partial \textbf{b}^1} = 
\frac{\partial L}{\partial \textbf{a}^2} 
\frac{\partial \textbf{a}^2}{\partial \textbf{z}^2}
\frac{\partial \textbf{z}^2}{\partial \textbf{a}^1}
\frac{\partial \textbf{a}^1}{\partial \textbf{z}^1}
\frac{\partial \textbf{z}^1}{\partial \textbf{b}^1},
$$
 where the new quantity is
$$
\frac{\partial \textbf{z}^1}{\partial \textbf{b}^1} = 
\left[
	\matrix{
		1 & 0 \\
		0 & 1
	}
\right]
$$
(see equation (3)).

All of the above gradients can be computed after randomly initializing the weights and biases of the network and after providing some random input vector $\textbf{a}^0$. To see the actual implementation of this test, I suggest to view the code in `tests/test_model.py`. 

# Empirical results

The implemented neural network was tested on the MNIST dataset containing hand-written digits from $0$ to $9$, which can be downloaded from [here](https://www.kaggle.com/jidhumohan/mnist-png). The dataset contains $60,000$ training and $10,000$ testing images with a height and width of 28 pixels each. We tested a couple of different model architectures and noticed that, in our implementation, deep neural networks suffered from the vanishing gradient problem very quickly, so we tried to keep the model architecture fairly simple. 

So, we chose a model with an input layer of $28 \times 28 = 784$ neurons (which was simply given by the image size), 32 neurons in the first hidden layer, 16 neurons in the second hidden layer and 10 neurons in the output layer. The two hidden layers both used the tanh activation function and the output layer used the softmax activation function while the weight updates were performed using Stochastic Gradient Descent. Finally, we compared the results of our model with the same network architecture implemented in Tensorflow. 

Below you can see a summary of the results and the settings that we used:

| Library    | Epochs | Batch Size | Learning Rate | Categorical Cross-Entropy | Accuracy | Precision | Recall | Training Time | Processing Unit |
| ---------- | ------ | ---------- | ------------- | ------------------------- | -------- | --------- | ------ | ------------- | --------------- |
| Our own    | 5      | 32         | 0.1           | 0.166                     | 0.990    | 0.950     | 0.950  | 17.95 min     | CPU             |
| Tensorflow | 5      | 32         | 0.1           | 0.290                     | 0.984    | 0.941     | 0.895  | 6.64 min      | GPU             |

Note that all metric values of categorical cross-entropy, accuracy, precision and recall were computed on the test set. Also notice that accuracy, precision and recall were computed using *micro averaging* which means that - independently of class - we count all instances of true positives (TP), false positives (FP), true negatives (TN) and false negatives (FN) and once that is done, accuracy, precision and recall are calculated as usual, i.e.:
$$
\text{accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

$$
\text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

$$
\text{recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

