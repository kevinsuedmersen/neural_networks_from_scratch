# Neural Networks from Scratch

This repository aims to derive and implement equations for training neural networks from scratch, which consist of an arbitrary number of fully connected, dense layers. First, we will attempt to derive equations for forward- as well as backward propagation in scalar form for a single training example. Then, we will extend these equations to a *matrix-based*  approach for a single training example, and finally, we will extend it to a matrix-based approach for processing `batch_size` examples at once. After implementing the necessary Python code, we will test the network's performance on the MNIST hand-written digits dataset and compare its performance with famous deep learning libraries such as TensorFlow.

# Table of contents

[TOC]

# Forward Propagation

In this section, we will start by explaining the forward pass, i.e. forward propagation, of a single training example. We will depict the computations in component form and then continue to explain how to vectorize the forward propagation for a single training example. We will show how the network generates predictions and how it evaluates the goodness of its predictions using a specific loss function. Finally, we will show a little "trick" to compute the forward propagation for `batch_size` training examples at once in a vectorized form.  

## Forward Propagation for a Single Training Example

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

Having computed $\textbf{a}^L$​, we can compute a certain *loss* which indicates how well or badly our model predicts for a *single* training example. For a classification problems where *exactly* one of  $n^L$​ classes must be predicted, a commonly used cost function is the *categorical cross entropy*, which is defined as follows


$$
L(\textbf{y}, \hat{\textbf{y}}) 
= -|| \textbf{y} \ log(\hat{\textbf{y}}) ||^2 
= -\sum_{i=1}^{n^L} y_i \ log(\hat{y}_i),
$$
where $\textbf{y}$ is the ground truth vector (containing the target values) where element $y_i = 1$ and all other elements are zero if the current training example belongs to class $i$, i.e. where $\textbf{y}$ is *one-hot-encoded*. 

In general, we want a loss function which has high values for bad predictions, i.e. when $\hat{y}_i$​​ is far away from $y_i$​​, and low values for good predictions, i.e. when $\hat{y}_i$​​ is very close to $y_i$​​. Let's see if component $i$​​ of (15) fulfills these requirements by considering the following example of a multi-class classificaton problem with 2 classes:

- Bad predictions

  - If $\textbf{y} = [1, 0]^T$ and $\hat{\textbf{y}} = [0, 1]^T$, then $L(y_i, \hat{y}_i)= -(1 \times log(0) + 0 \times log(1)) = -(-\infty + 0) = \infty$
  - If $\textbf{y} = [0, 1]^T$ and $\hat{\textbf{y}} = [1, 0]^T$, then $L(y_i, \hat{y}_i)= -(0 \times log(1) + 1 \times log(0)) = -(0 -\infty) = \infty$

- Good predictions

  - If $\textbf{y} = [1, 1]^T$ and $\hat{\textbf{y}} = [1, 1]^T$, then $L(y_i, \hat{y}_i)= -(1 \times log(1) + 1 \times log(1)) = -(0 + 0) = 0$
- If $\textbf{y} = [0, 0]^T$ and $\hat{\textbf{y}} = [0, 0]^T$, then $L(y_i, \hat{y}_i)= -(0 \times log(0) + 0 \times log(0)) = -(0 + 0) = 0$

in all of the 4 above cases, we get the desired result. 

## Forward Propagation for a Batch of  Training Examples

We will conclude this section by showing how to compute the complete forward propagation for `batch_size` training examples at once. It starts by defining your input data matrix as follows
$$
\textbf{X} = \textbf{A}^0,
$$
where the feature vectors of each training examples are stacked horizontally next to each other in a column-wise fashion. Assuming that we have $M$ training examples in our current batch and $n^0$ input features, equation (16) can be written out explicitly as
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

where $x_i^m = a_i^{0, m}$ both refer to the value of the i-th feature of the m-th training example. 

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
\right],
$$

where each element $m = 1, 2, ..., M$ of the above loss vector represents the loss we have already defined in equation (15), i.e.
$$
L(\textbf{y}^m, \hat{\textbf{y}}^m) = -\sum_{i=1}^{n^L} y_i^m log(\hat{y}_i^m).
$$

Having computed the loss vector $L(\textbf{Y}, \hat{\textbf{Y}})$, we can now aggregate over all $M$ training examples to compute a certain *cost*, which is just the average over all `batch_size` training examples
$$
C = \frac{1}{M} \sum_{m=1}^M L(\textbf{y}^m, \hat{\textbf{y}}^m) 
= -\frac{1}{M} \sum_{m=1}^M \sum_{i=1}^{n^L} y_i^m log(\hat{y}_i^m),
$$
Note that the loss function represents an error over a *single* training example, while the cost function is an aggregation of the loss over $M$​​​ training examples. When computing the cost for $M$​​​ training examples, it makes sense to choose the average as an aggregation, because the average cost doesn't increase linearly with the `batch_size`. Also, the cost function may include a regularization term, which should be monotonically increasing in the number of parameters of the model, to account for overfitting.

# Backward Propagation

Neural networks learn by iteratively adjusting their weights and biases such that the cost decreases, i.e. such that the predictions become more accurate. This goal is achieved by (1) computing all partial derivatives of the cost w.r.t. the weights and biases in the network (the *gradient*) and (2) by updating the weights and biases using *gradient descent*. The next sections will start by describing the backward propagation for a single training example and then extend this procedure for `batch_size` training examples at once. 

The backpropagation algorithm works as follows. For any given layer $l$​​​, the backpropagation algorithm computes an intermediate quantity, the so called *error*​​​ at layer $l$, and then computes the gradients of the weights and biases in layer $l$ using that error. Then, the error is propagated one layer backwards and the gradients are computed again. This process is repeated recursively until the gradients of the weights and biases in layer 1 (layer with index 1) are computed. 

The backpropagation algorithm is based on 4 key equations which we will derive in detail in the following sections. The four key equations are as follows:

- BP1.x: An equation for the error at the output layer, needed for initializing the backpropagation algorithm
  - When considering a single training example, we will refer to this equation as $\boldsymbol{\delta}^L$ or BP1.1
  - When considering `batch_size`training examples, we will refer to this equation as $\boldsymbol{\Delta}^L$​ or BP1.2​, which is a matrix where each column contains the error for a different training example.
- BP2.x: A recursive equation relating the error at layer $l+1$​​​ to the error at layer $l$​​​​​​​​, needed for recursively calculating the error at each layer.
  - When considering a single training example, we will refer to this equation as $\boldsymbol{\delta}^l$
  - When considering `batch_size` training examples, we will refer to this equation as $\boldsymbol{\Delta}^l$​, which is a matrix where each column contains the error for a different training example.
  - Note that in the first iteration, we must set $\boldsymbol{\delta}^l = \boldsymbol{\delta}^L$​​ or $\boldsymbol{\Delta}^l = \boldsymbol{\Delta}^L$ which we already computed in BP1.1 and BP1.2 respectively. ​​After that, we can recursively substitute the error all the way back to layer index $0$. 
- BP3.x: An equation relating the error at layer $l$ to:
  - The derivative of the *loss* function w.r.t the weights in layer $l$​ when considering a *single* training example, i.e. $\frac{\partial L}{\partial \textbf{W}^l}$​. We'll refer to this equation as BP3.1
  - The derivative of the *cost* function w.r.t. the weights in layer $l$​ when considering a *batch* of training examples, i.e. $\frac{\partial C}{\partial \textbf{W}^l}$​​.We'll refer to this equation as BP3.2
- BP4.x: An equation relating the error at layer $l$ to:
  - The derivative of the *loss* function w.r.t the biases in layer $l$​​​ when considering a *single* training example, i.e. $\frac{\partial L}{\partial \textbf{b}^l}$​​​. We'll refer to this equation as BP4.1
  - The derivative of the *cost* function w.r.t. the biases in layer $l$​​​​ when considering a *batch* of training examples, i.e. $\frac{\partial C}{\partial \textbf{b}^l}$​​​​​​.We'll refer to this equation as BP4.2

Most of the work will go into deriving equations BP1.1-BP4.1 and applying these equations to `batch_size` equations at once is just a little overhead in math but will save a lot of time when running the actual code.  

## Backpropagation for a Single Training Example

### BP1.1

The error at the output layer $\boldsymbol{\delta}^L$​​​, which represents $\frac{\partial L}{\partial \textbf{z}^L}$​​, can be expressed as follows (remembering the chain rule from calculus)
$$
\boldsymbol{\delta}^L \coloneqq \frac{\partial L}{\partial \textbf{z}^L} 
= \frac{\partial L}{\partial \textbf{a}^L} \frac{\partial \textbf{a}^L}{\partial \textbf{z}^L}
= \nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L).
$$

Remember that the derivative of a function yielding a scalar, the loss function, w.r.t. a vector is defined as a row vector, i.e. the *gradient* $\nabla L(\textbf{a}^L)$, and that the derivative of a function yielding a vector w.r.t. another vector is defined as a matrix, i.e. the *Jacobi* matrix $\textbf{J}_{\textbf{a}^L}(\textbf{z}^L)$. So, writing out every component of the above expression produces
$$
\nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L) = 
\left[
	\matrix{
    \frac{\partial L}{\partial a^L_1}, & \frac{\partial L}{\partial a^L_2}, & ..., & \frac{\partial L}{\partial a^L_{n^L}}
    }
\right]
\left[
	\matrix{
    	\frac{\partial a^L_1}{\partial z^L_1}, & \frac{\partial a^L_1}{\partial z^L_2}, & ..., & \frac{\partial a^L_1}{\partial z^L_{n^L}} \\
        \frac{\partial a^L_2}{\partial z^L_1}, & \frac{\partial a^L_2}{\partial z^L_2}, & ..., & \frac{\partial a^L_2}{\partial z^L_{n^L}} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{\partial a^L_{n^L}}{\partial z^L_1}, & \frac{\partial a^L_{n^L}}{\partial z^L_2}, & ..., & \frac{\partial a^L_{n^L}}{\partial z^L_{n^L}}
    }
\right],
$$
which ends up as a $1 \times n^L$ row vector. In its most general form, the above vector matrix product represents BP1.1, so without making any assumptions about the loss function $L$​​ and the activation function in the output layer $a^L$​​, the above equation cannot be simplified any further. In the next section, we will show how to simplify this equation by choosing a specific loss and activation function. 

#### Example

For multi-class classification problems, a common choice for the loss function is the categorical cross entropy (see equation 15) and a common choice for the activation function in the output layer is the *softmax* function, which unlike e.g. the sigmoid activation function, takes a vector as input and also outputs a vector, whose $j$​-th component is defined as follows 
$$
a^l_j = f([z^l_1, z^l_2, ..., z^l_j, ..., z^l_{n^l}])_j = \frac{e^{z^l_j}}{\sum_{k=1}^{n^l} e^{z^l_k}}.
$$
First, we will try to find concrete expressions for each component of $\nabla L(\textbf{a}^L)$ in (27). From the categorical cross entropy loss function in equation 15, we can derive that for $i = 1, 2, ..., n^L$,
$$
\frac{\partial L}{\partial a^{L}_j} = - \frac{y_j}{a^{L}_j},
$$

where we used the fact that $\hat{y}_j = a^{L}_j$​​​​​​​. 

Second, we want to find concrete expressions for each component of $\textbf{J}_{\textbf{a}^L}(\textbf{z}^L)$​​​​ in (26). When taking the derivative of the Softmax function, we need to consider two cases. The first case is represented by $\frac{\partial a^L_j}{\partial z^L_k}$​​​​, if $j=k$​​​​, i.e. $\frac{\partial a^L_j}{\partial z^L_j}$​​​​.
$$
\large{
\begin{array}{l}
	\frac{\partial a^L_j}{\partial z^L_j}
	= \frac{e^{z^L_j} \left( \sum^{n^L}_{k=1} e^{z^l_k} \right) - e^{z^L_j} e^{z^L_j}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)^2} \\
	= \frac{e^{z^L_j} \left( \left(\sum^{n^L}_{k=1} e^{z^L_k} \right) - e^{z^L_j} \right)}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)^2} \\
	= \frac{e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \frac{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right) - e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \\
	= \frac{e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \left( \frac{\sum^{n^L}_{k=1} e^{z^L_k}}{\sum^{n^L}_{k=1} e^{z^L_k}} - \frac{e^{z^L_k}}{\sum^{n^L}_{k=1} e^{z^L_k}} \right) \\ 
	= \frac{e^{z^L_j}}{\sum^{n^L}_{k=1} e^{z^L_k}} \left( 1 - \frac{e^{z^L_k}}{\sum^{n^L}_{k=1} e^{z^L_k}} \right),
\end{array}
}
$$
 where we can now use the definition of the Softmax function (equation 27) again to simplify further to
$$
\frac{\partial a^L_j}{\partial z^L_j} = a^L_j (1 - a^L_j).
$$
The second case is represented by $\frac{\partial a^L_j}{\partial z^L_k}$​​, where $k \neq j$​​, so that
$$
\large{
	\begin{array}{l}
		\frac{\partial a^L_j}{\partial z^L_k}
		= \frac{0 \times \left(\sum^{n^L}_{k=1} e^{z^L_k} \right) - e^{z^L_j} e^{z^L_k}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)^2} \\
		= \frac{- e^{z^L_j} e^{z^L_k}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)^2} \\
		= - \frac{e^{z^L_j}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)} \frac{e^{z^L_k}}{\left( \sum^{n^L}_{k=1} e^{z^L_k} \right)} \\
		= -a^L_j \ a^L_k.
	\end{array}
}
$$
So, summarizing, 
$$
\frac{\partial a^L_j}{\partial z^L_k} = 
\large{
	\begin{cases}
		a^L_j(1-a^L_j) & \text{if} & j=k \\
        -a^L_j \ a^L_k & \text{if} & j \neq k
	\end{cases}
}
$$

Using above expression, we can now fill in each component of (26) as follows
$$
\nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L) = 
- \left[
	\matrix{
    	\frac{y_1}{a^L_1}, & \frac{y_2}{a^L_2}, & ..., & \frac{y_{n^L}}{a^L_{n^L}}
    }
\right]
\left[
	\matrix{
    	a^L_1 (1 - a^L_1), & -a^L_1 \ a^L_2, & ..., & -a^L_1 \ a^L_{n^L} \\
        -a^L_2 \ a^L_1, & a^L_2 (1 - a^L_2), & ..., & -a^L_2 \ a^L_{n^L} \\
        \vdots & \vdots & \ddots & \vdots \\
        -a^L_{n^L} \ a^L_1, & -a^L_{n^L} \ a^L_2, & ..., & a^L_{n^L} (1 - a^L_{n^L})
    }
\right],
$$

which, when multiplied out, yields
$$
\nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L) = 
\left[
	\matrix{
		-y_1 + a^L_1(y_1 + y_2 + ... + y_{n^L}), 
		& -y_2 + a^L_2(y_1 + y_2 + ... + y_{n^L}), 
		& ..., 
		& -y_{n^L} + a^L_{n^L}(y_1 + y_2 + ... + y_{n^L}) 
	}
\right].
$$
 Notice that $(y_1 + y_2 + ... + y_{n^L}) = 1$ always due to the one hot encoded target vector $\textbf{y}$. So, we can simplify the above expression to
$$
\nabla L(\textbf{a}^L) \ \textbf{J}_{\textbf{a}^L}(\textbf{z}^L) = 
- \left[
	\matrix{
		(y_1 - a^L_1), & (y_2 - a^L_2), ..., & (y_{n^L} - a^L_{n^L})
	}
\right].
$$


### BP2.1

In order to represent the error of the previous layer $\boldsymbol{\delta}^{l-1}$​ in terms of the error in the current layer $\boldsymbol{\delta}^{l}$, it helps to view the loss function as a nested functions of weighted input vectors, i.e. $L(\textbf{z}^l(\textbf{z}^{l-1}))$ which we want to derive w.r.t. $\textbf{z}^{l-1}$. This can be done as follows
$$
\boldsymbol{\delta}^{l-1} \coloneqq 
\frac{\partial L}{\partial \textbf{z}^{l-1}}
= \frac{\partial L}{\partial \textbf{z}^l} \frac{\partial \textbf{z}^l}{\partial \textbf{z}^{l-1}}
= \nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}),
$$
which can be written out explicitly as
$$
\nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}) 
= \left[
	\matrix{
		\frac{\partial L}{\partial z^l_1}, & \frac{\partial L}{\partial z^l_2}, & ..., & \frac{\partial L}{\partial z^l_{n^l}} 
	}
\right]
\left[
	\matrix{
		\frac{\partial z^l_1}{\partial z^{l-1}_1}, & \frac{\partial z^l_1}{\partial z^{l-1}_2}, & ..., & \frac{\partial z^l_1}{\partial z^{l-1}_{n^{l-1}}} \\
        \frac{\partial z^l_2}{\partial z^{l-1}_1}, & \frac{\partial z^l_2}{\partial z^{l-1}_2}, & ..., & \frac{\partial z^l_2}{\partial z^{l-1}_{n^{l-1}}} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{\partial z^l_{n^l}}{\partial z^{l-1}_1}, & \frac{\partial z^l_{n^l}}{\partial z^{l-1}_2}, & ..., & \frac{\partial z^l_{n^l}}{\partial z^{l-1}_{n^{l-1}}}
	}
\right].
$$
Again, without making explicit assumptions about the loss function and the activation function in layer $l-1$, we cannot simplify the above expression any further. 

#### Example

To work out (37), we will assume that we are using the categorical cross entropy loss function and that the activation function $f$ in layer $l-1$ is a function taking a scalar input and outputting a scalar as well, such as the sigmoid, tanh or ReLU functions which are common activation functions in the hidden layers. 

In order to find an expression for every component of $\nabla L(\textbf{z}^l)$, notice that by definition, 
$$
\frac{\partial L}{\partial z^l_j} = \delta^l_j.
$$
 In order to find an expression for every component of $\textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1})$, recall that
$$
\large
\begin{array}{l}
	z^l_j & = \sum^{n^{l-1}}_{k=1} \left( a^{l-1}_k w^l_{j, k} \right) + b^l_j \\
	& = \sum^{n^{l-1}}_{k=1} \left( f(z^{l-1}_k) \ w^l_{j, k} \right) + b^l_j
\end{array}
$$
and therefore, 
$$
\frac{\partial z^l_j}{\partial z^{l-1}_k} = w^l_{j, k} \ f'(z^{l-1}_k).
$$
Using (38) and (40), we can fill in each component of (37) as follows
$$
\nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}) 
= \left[
	\matrix{
		 \delta^l_1, & \delta^l_2, & ..., & \delta^l_{n^l}
	}
\right]
\left[
	\matrix{
		w^l_{1, 1} \ f'(z^{l-1}_1), & w^l_{1, 2} \ f'(z^{l-1}_2), & ..., & w^l_{1, n^{l-1}} \ f'(z^{l-1}_{n^{l-1}}) \\
		w^l_{2, 1} \ f'(z^{l-1}_1), & w^l_{2, 2} \ f'(z^{l-1}_2), & ..., & w^l_{2, n^{l-1}} \ f'(z^{l-1}_{n^{l-1}}) \\
		\vdots & \vdots & \ddots & \vdots \\
		w^l_{n^l, 1} \ f'(z^{l-1}_1), & w^l_{n^l, 2} \ f'(z^{l-1}_2), & ..., & w^l_{n^l, n^{l-1}} \ f'(z^{l-1}_{n^{l-1}})
	}
\right],
$$
which can be decomposed into
$$
\nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1}) 
= \left[
	\matrix{
		 \delta^l_1, & \delta^l_2, & ..., & \delta^l_{n^l}
	}
\right]
\left[
	\matrix{
		w^l_{1, 1}, & w^l_{1, 2}, & ..., & w^l_{1, n^{l-1}} \\
		w^l_{2, 1}, & w^l_{2, 2}, & ..., & w^l_{2, n^{l-1}} \\
		\vdots & \vdots & \ddots & \vdots \\
		w^l_{n^l, 1}, & w^l_{n^l, 2}, & ..., & w^l_{n^l, n^{l-1}}
	}
\right]
\odot
\left[
	\matrix{f(z^{l-1}_1), & f(z^{l-1}_2), & ..., & f(z^{l-1}_{n^{l-1}})
	}
\right],
$$
or equivalently, 
$$
\boldsymbol{\delta}^{l-1} = (\boldsymbol{\delta}^l)^T \ \textbf{W}^l \odot f(\textbf{z}^{l-1})^T,
$$
where we used $\boldsymbol{\delta}^{l-1} \coloneqq \nabla L(\textbf{z}^l) \ \textbf{J}_{\textbf{z}^l}(\textbf{z}^{l-1})$ to clarify that we ended up with a recursive equation. 

### BP3.1

After calculating the errors for each layer, we now want to relate them to the derivative of the loss w.r.t the weights (and later the biases as well). To find an expression for the derivative of the loss w.r.t. the weights in layer $l$, it might help to view weight matrix $\textbf{W}^l$ as a long, flattened vector $\textbf{w}^l$, whose, $(i,j)$-th element represents (or came from) the weight in the $i$-th row and $j$-th column of $\textbf{W}^l$, such that $\textbf{W}^l[i, j] = \textbf{w}^l[(i,j)]$​​. 

With that in mind, we can express the derivative of the loss w.r.t the weights in layer $l$ as
$$
\frac{\partial L}{\partial \textbf{w}^l} 
= \frac{\partial L}{\partial \textbf{z}^l} \frac{\partial \textbf{z}^l}{\partial \textbf{w}^l}
= \nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{w}^l),
$$
which can be written out explicitly as
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{w}^l) = 
\left[
	\matrix{
    	\frac{\partial L}{\partial z^l_1}, & \frac{\partial L}{\partial z^l_2}, & ..., & \frac{\partial L}{\partial z^l_{n^l}}
        }
\right]
\left[
	\matrix{
    	\frac{\partial z^l_1}{\partial w^l_{1, 1}}, & \frac{\partial z^l_1}{\partial w^l_{1, 2}}, & ..., & \frac{\partial z^l_1}{\partial w^l_{1, n^{l-1}}}, 
        & \frac{\partial z^l_1}{\partial w^l_{2, 1}}, & \frac{\partial z^l_1}{\partial w^l_{2, 2}}, & ..., & \frac{\partial z^l_1}{\partial w^l_{2, n^{l-1}}}, 
        & ..., & 
        \frac{\partial z^l_1}{\partial w^l_{n^l, 1}}, & \frac{\partial z^l_1}{\partial w^l_{n^l, 2}}, & ..., & \frac{\partial z^l_1}{\partial w^l_{n^l, n^{l-1}}} \\

		\frac{\partial z^l_2}{\partial w^l_{1, 1}}, & \frac{\partial z^l_2}{\partial w^l_{1, 2}}, & ..., & \frac{\partial z^l_2}{\partial w^l_{1, n^{l-1}}}, 
        & \frac{\partial z^l_2}{\partial w^l_{2, 1}}, & \frac{\partial z^l_2}{\partial w^l_{2, 2}}, & ..., & \frac{\partial z^l_2}{\partial w^l_{2, n^{l-1}}}, 
        & ..., & 
        \frac{\partial z^l_2}{\partial w^l_{n^l, 1}}, & \frac{\partial z^l_2}{\partial w^l_{n^l, 2}}, & ..., & \frac{\partial z^l_2}{\partial w^l_{n^l, n^{l-1}}} \\

		\vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\

		\frac{\partial z^l_{n^l}}{\partial w^l_{1, 1}}, & \frac{\partial z^l_{n^l}}{\partial w^l_{1, 2}}, & ..., & \frac{\partial z^l_{n^l}}{\partial w^l_{1, n^{l-1}}}, 
        & \frac{\partial z^l_{n^l}}{\partial w^l_{2, 1}}, & \frac{\partial z^l_{n^l}}{\partial w^l_{2, 2}}, & ..., & \frac{\partial z^l_{n^l}}{\partial w^l_{2, n^{l-1}}}, 
        & ..., & 
        \frac{\partial z^l_{n^l}}{\partial w^l_{n^l, 1}}, & \frac{\partial z^l_{n^l}}{\partial w^l_{n^l, 2}}, & ..., & \frac{\partial z^l_{n^l}}{\partial w^l_{n^l, n^{l-1}}}
    }
\right].
$$

Again, we will first find expressions for each component of $\nabla L(\textbf{z}^l)$ and after that, for each component of $J_{\textbf{z}^l}(\textbf{w}^l)$. Deriving the components of $\nabla L(\textbf{z}^l)$ is easy, because per definition, 
$$
\frac{\partial L}{\partial z^l_j} \coloneqq \delta^l_j.
$$
To derive each component of $J_{\textbf{z}^l}(\textbf{w}^l)$, we need to consider two cases again. First, consider $\frac{\partial z^l_j}{\partial w^l_{j, k}}$ if $j = k$. Remember that $z^l_j = \sum^{n^{l-1}}_{k=1} w^l_{j,k} \ a^{l-1}_k + b^l_j$, so
$$
\frac{\partial z^l_j}{\partial w^l_{j,k}} = a^{l-1}_k.
$$
Next, consider $\frac{\partial z^l_j}{\partial w^l_{j, k}}$ if $j \neq k$. In that case, the weight $w^l_{j, k}$ is not connected to neuron $j$ in layer $l$, so 
$$
\frac{\partial z^l_j}{\partial w^l_{j,k}} = 0.
$$
Summarizing, we have that 
$$
\large
\frac{\partial z^l_j}{\partial w^l_{j,k}} =
\begin{cases}
	a^{l-1}_k & \text{if} \ j=k \\
	0 & \text{if} \ j \neq k \\
\end{cases}
$$
Using (46) and (49), we can now fill in each value of (45) as follows
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{w}^l) = 
\left[
	\matrix{
    	\delta^l_1, & \delta^l_2, & ..., & \delta^l_{n^l}
        }
\right]
\left[
	\matrix{
    	a^{l-1}_1, & a^{l-1}_2, & ..., & a^{l-1}_{n^{l-1}}, 
        & 0, & 0, & ..., & 0, 
        & ..., & 
        0, & 0, & ..., & 0 \\

		0, & 0, & ..., & 0, 
        & a^{l-1}_1, & a^{l-1}_2, & ..., & a^{l-1}_{n^{l-1}}, 
        & ..., & 
        0, & 0, & ..., & 0 \\

		\vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\

		0, & 0, & ..., & 0, 
        & 0, & 0, & ..., & 0, 
        & ..., 
        & a^{l-1}_1, & a^{l-1}_2, & ..., & a^{l-1}_{n^{l-1}} 
    }
\right].
$$
Multiplying out the above expression yields the following $1 \times (n^l \times n^{l-1})$ row vector
$$
\nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{w}^l)
= \left[
	\matrix{
		\delta^l_1 \ a^{l-1}_1, & \delta^l_1 \ a^{l-1}_2, & ..., & \delta^l_1 \ a^{l-1}_{n^{l-1}}, &
		\delta^l_2 \ a^{l-1}_1, & \delta^l_2 \ a^{l-1}_2, & ..., & \delta^l_2 \ a^{l-1}_{n^{l-1}}, &
		... &
		\delta^l_{n^l} \ a^{l-1}_1, & \delta^l_{n^l} \ a^{l-1}_2, & ..., & \delta^l_{n^l} \ a^{l-1}_{n^{l-1}}, &
	}
\right]
$$
Assume that we wanted to represent the above $1 \times (n^l \times n^{l-1})$ row vector into a $n^l \times n^{l-1}$ matrix. Then, we could represent the above equation as
$$
\frac{\partial L}{\partial \textbf{w}^l} = 
\text{flatten} \left( \left[
	\matrix{
		\delta^l_1 a^{l-1}_1, & \delta^l_1 a^{l-1}_2, & ... & \delta^l_1 a^{l-1}_{n^{l-1}} \\
		\delta^l_2 a^{l-1}_1, & \delta^l_2 a^{l-1}_2, & ... & \delta^l_2 a^{l-1}_{n^{l-1}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\delta^l_{n^l} a^{l-1}_1, & \delta^l_{n^l} a^{l-1}_2, & ... & \delta^l_{n^l} a^{l-1}_{n^{l-1}} \\
	}
\right] \right),
$$
where $\text{flatten}$ is a function that flattens any 2 dimensional matrix into a 1 dimensional row vector row-wise. With (51), we can now see that we can decompose it into the following vector by vector multiplication
$$
\frac{\partial L}{\partial \textbf{w}^l} = 
\text{flatten} \left( \left[
	\matrix{
		\delta^l_1 \\
		\delta^l_2 \\
		\vdots \\
		\delta^l_{n^l} \\
	}
\right]
\left[
	\matrix{
		a^{l-1}_1, & a^{l-1}_2, & ..., & a^{l-1}_{n^{l-1}} 
	}
\right] \right) 
= \text{flatten} \left( \boldsymbol{\delta}^l \ (\textbf{a}^{l-1})^T \right).
$$


The above equation represents BP3.1. 

### BP4.1 

Now, we want to relate the errors of each layer to the derivative of the loss w.r.t. the biases. The derivative of the loss w.r.t the biases can be expressed as follows
$$
\frac{\partial L}{\partial \textbf{b}^l} 
= \frac{\partial L}{\partial \textbf{z}^l} \frac{\partial \textbf{z}^l}{\partial \textbf{b}^l} 
= \nabla L(\textbf{z}^l) \ J_{\textbf{z}^l}(\textbf{b}^l),
$$
which can be written out explicitly as
$$
\left[
	\matrix{
		\frac{\partial L}{\partial z^l_1}, & \frac{\partial L}{\partial z^l_2}, & ..., & \frac{\partial L}{\partial z^l_{n^l}}
	}
\right]
\left[
	\matrix{
		\frac{\partial z^l_1}{\partial b^l_1}, & \frac{\partial z^l_1}{\partial b^l_2}, & ..., & \frac{\partial z^l_1}{\partial b^l_{n^l}} \\
		\frac{\partial z^l_2}{\partial b^l_1}, & \frac{\partial z^l_2}{\partial b^l_2}, & ..., & \frac{\partial z^l_2}{\partial b^l_{n^l}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\frac{\partial z^l_{n^l}}{\partial b^l_1}, & \frac{\partial z^l_{n^l}}{\partial b^l_2}, & ..., & \frac{\partial z^l_{n^l}}{\partial b^l_{n^l}} \\
	}
\right]
$$


which we can multiply out as follows
$$
\frac{\partial L}{\partial \textbf{b}^l} = \left[
	\matrix{
		\frac{\partial L}{\partial z^l_1} \frac{\partial z^l_1}{\partial b^l_1} +
        \frac{\partial L}{\partial z^l_2} \frac{\partial z^l_2}{\partial b^l_1} + 
        ... + 
        \frac{\partial L}{\partial z^l_{n^l}} \frac{\partial z^l_{n^l}}{\partial b^l_1}, &
        
        \frac{\partial L}{\partial z^l_1} \frac{\partial z^l_1}{\partial b^l_2} +
        \frac{\partial L}{\partial z^l_2} \frac{\partial z^l_2}{\partial b^l_2} + 
        ... + 
        \frac{\partial L}{\partial z^l_{n^l}} \frac{\partial z^l_{n^l}}{\partial b^l_2}, &
        
        ..., &
        
        \frac{\partial L}{\partial z^l_1} \frac{\partial z^l_1}{\partial b^l_{n^l}} +
        \frac{\partial L}{\partial z^l_2} \frac{\partial z^l_2}{\partial b^l_{n^l}} + 
        ... + 
        \frac{\partial L}{\partial z^l_{n^l}} \frac{\partial z^l_{n^l}}{\partial b^l_{n^l}}
	}
\right],
$$
which can be simplified to
$$
\frac{\partial L}{\partial \textbf{b}^l} = 
\left[
	\matrix{
		\sum^{n^l}_{j=1} \frac{\partial L}{\partial z^l_j} \frac{\partial z^l_j}{\partial b^l_1}, &
		\sum^{n^l}_{j=1} \frac{\partial L}{\partial z^l_j} \frac{\partial z^l_j}{\partial b^l_2}, &
		..., &
		\sum^{n^l}_{j=1} \frac{\partial L}{\partial z^l_j} \frac{\partial z^l_j}{\partial b^l_{n^l}}
	}
\right].
$$
Notice that $\frac{\partial z^l_j}{\partial b^l_k} = 0$ if $j \neq k$, so we can simplify the above expression to 
$$
\frac{\partial L}{\partial \textbf{b}^l} =
\left[
	\matrix{
		\frac{\partial L}{\partial z^l_1} \frac{\partial z^l_1}{\partial b^l_1}, &
		\frac{\partial L}{\partial z^l_2} \frac{\partial z^l_2}{\partial b^l_2}, &
		..., &
		\frac{\partial L}{\partial z^l_{n^l}} \frac{\partial z^l_{n^l}}{\partial b^l_{n^l}}
	}
\right].
$$




Furthermore, notice that we have already defined $\frac{\partial L}{\partial z^l_j} \coloneqq \delta^l_j$​ and notice that since, $z^l_j = \sum^{n^{l-1}}_{k=1} w^l_{j, k} \ a^{l-1}_k + b^l_j$​, we know that $\frac{\partial z^l_j}{\partial b^l_j} = 1$​​. Using these insights, we can simplify equation (59) to
$$
\frac{\partial L}{\partial \textbf{b}^l} =
\left[
	\matrix{
		\delta^l_1 , & \delta^l_2, & ..., & \delta^l_{n^l}
	}
\right] =
\left( \boldsymbol{\delta}^l \right)^T.
$$
The above equation represents BP4.1.



## Backpropagation for batch_size Training Examples at once

In the previous section, we have derived equations which can help us to compute the gradients of a single training example. Computing these expressions separately for each training example will take a tremendous amount of time, so in this section, we aim to extend these equations so that we can compute the gradient for `batch_size` training examples at once, harnessing already optimized and extremely efficient matrix multiplication libraries such as `numpy`. 

### BP1.2

Recall from BP1.1 that 
$$
\boldsymbol{\delta}^L = \frac{\partial L}{\partial \textbf{a}^L} \odot \left( f'(\textbf{z}^L) \right)^T.
$$
We want to end up with an expression where each column represents a different training example. In order to achieve that, we will redefine the above expression as its transpose like so
$$
\begin{array}{c}
\boldsymbol{\delta}^L \coloneqq 
\left( \boldsymbol{\delta}^L \right)^T \\
= \left(\frac{\partial L}{\partial \textbf{a}^L} \right)^T \odot f'(\textbf{z}^L) \\
 = \left[
	\matrix{
		\frac{\partial L}{\partial a^L_1} \\
		\frac{\partial L}{\partial a^L_2} \\
		\vdots \\
		\frac{\partial L}{\partial a^L_{n^L}}
	}
\right]
\odot
\left[
	\matrix{
		f'(z^L_1) \\
		f'(z^L_2) \\
		\vdots \\
		f'(z^L_{n^L})
	}
\right]
\end{array}.
$$
To calculate $\boldsymbol{\delta}^L$​ for $m = 1, 2, ..., M$​ training examples at once, stack each $\boldsymbol{\delta}^{L, m}$​ next to each other in a column wise fashion like this
$$
\begin{array}{c}
	\boldsymbol{\Delta}^L \coloneqq 
	\left[
		\matrix{\boldsymbol{\delta}^{L, 1}, & \boldsymbol{\delta}^{L, 2}, & ..., & \boldsymbol{\delta}^{L, M}}
	\right] \\
	= \left[
		\matrix{
			\frac{\partial L}{\partial a^{L, 1}_1}, & \frac{\partial L}{\partial a^{L, 2}_1}, & ..., & \frac{\partial L}{\partial a^{L, M}_1} \\
			\frac{\partial L}{\partial a^{L, 1}_2}, & \frac{\partial L}{\partial a^{L, 2}_2}, & ..., & \frac{\partial L}{\partial a^{L, M}_2} \\
			\vdots & \vdots & \ddots & \vdots \\
			\frac{\partial L}{\partial a^{L, 1}_{n^L}}, & \frac{\partial L}{\partial a^{L, 2}_{n^L}}, & ..., & \frac{\partial L}{\partial a^{L, M}_{n^L}}
		}
	\right]
	\odot
	\left[
		\matrix{
			f'(z^{L, 1}_1), & f'(z^{L, 2}_1), & ..., & f'(z^{L, M}_1) \\
			f'(z^{L, 1}_2), & f'(z^{L, 2}_2), & ..., & f'(z^{L, M}_2) \\
			\vdots & \vdots & \ddots & \vdots \\
			f'(z^{L, 1}_{n^L}), & f'(z^{L, 2}_{n^L}), & ..., & f'(z^{L, M}_{n^L})
		}
	\right]
\end{array},
$$
or written more compactly as
$$
\boldsymbol{\Delta}^L = \frac{\partial L}{\partial \textbf{A}^L} \odot f'(\textbf{Z}^L),
$$

which represents BP1.2.

#### Example

Like in the example of BP1.2, we will assume that we use the categorical cross entropy loss function and the sigmoid activation function. Then, using equation (35), we can explicitly write out equation (66) as follows
$$
\boldsymbol{\Delta}^L = - \left[
	\matrix{
		\left( y_1^1 - a^{L, 1}_1 \right), & \left( y_1^2 - a^{L, 2}_1 \right), & ..., & \left( y_1^M - a^{L, M}_1 \right) \\
		\left( y_2^1 - a^{L, 1}_2 \right), & \left( y_2^2 - a^{L, 2}_2 \right), & ..., & \left( y_2^M - a^{L, M}_2 \right) \\
		\vdots & \vdots & \ddots & \vdots \\
		\left( y_{n^L}^1 - a^{L, 1}_{n^L} \right), & \left( y_{n^L}^2 - a^{L, 2}_{n^L} \right), & ..., & \left( y_{n^L}^M - a^{L, M}_{n^L} \right)
	}
\right]
= \textbf{Y} - \textbf{A}^L,
$$

which is easily computed, because we are given $\textbf{Y}$ (since we are talking about a supervised learning problem here) and we already computed $\textbf{A}^L$ during the forward propagation. 

### BP2.2

Recall from BP2.1 that 
$$
\boldsymbol{\delta}^{l-1} = \left( f'(\textbf{z}^{l-1}) \right)^T \odot (\boldsymbol{\delta}^l)^T \textbf{W}^l.
$$
Again, we want to end up with an expression, where each column represents a different training example, so we will redefine the above expression as its transpose like so
$$
\boldsymbol{\delta}^{l-1} \coloneqq \left( \boldsymbol{\delta}^{l-1} \right)^T = f'(\textbf{z}^{l-1}) \odot \left( \textbf{W}^l \right)^T \boldsymbol{\delta}^l,
$$
remembering that when transposing a matrix multiplication, we have to transpose each factor and reverse their orders (except for the Hadamard product, because it's an element-by-element multiplication). So, for each $1, 2, ...M$, we will stack (column-wise) the errors and weighted inputs of each training example as follows
$$
\left[
	\matrix{
		\boldsymbol{\delta}^{l-1, 1}, & \boldsymbol{\delta}^{l-1, 2}, & ..., & \boldsymbol{\delta}^{l-1, M}
	}
\right] =
\left[
	\matrix{
		f'(\textbf{z}^{l-1, 1}), & f'(\textbf{z}^{l-1, 2}), & ..., & f'(\textbf{z}^{l-1, M}) 
	}
\right] \odot
\left( \textbf{W}^l \right)^T
\left[
	\matrix{
		\boldsymbol{\delta}^{l, 1}, & \boldsymbol{\delta}^{l, 2}, & ..., & \boldsymbol{\delta}^{l, M}
	}
\right].
$$


Writing out each element of the above equation, we will get
$$
\left[
	\matrix{
		\delta^{l-1, 1}_1, & \delta^{l-1, 2}_1, & ..., & \delta^{l-1, M}_1 \\
		\delta^{l-1, 1}_2, & \delta^{l-1, 2}_2, & ..., & \delta^{l-1, M}_2 \\
		\vdots & \vdots & \ddots & \vdots \\
		\delta^{l-1, 1}_{n^{l-1}}, & \delta^{l-1, 2}_{n^{l-1}}, & ..., & \delta^{l-1, M}_{n^{l-1}} \\
	}
\right] = 
\left[
	\matrix{
		f'(z^{l-1, 1}_1), & f'(z^{l-1, 2}_1), & ..., & f'(z^{l-1, M}_1) \\
		f'(z^{l-1, 1}_2), & f'(z^{l-1, 2}_2), & ..., & f'(z^{l-1, M}_2) \\
		\vdots & \vdots & \ddots & \vdots \\
		f'(z^{l-1, 1}_{n^{l-1}}), & f'(z^{l-1, 2}_{n^{l-1}}), & ..., & f'(z^{l-1, M}_{n^{l-1}})
	}
\right] \odot
\left[
	\matrix{
		w^l_{1,1}, & w^l_{2,1}, & ..., & w^l_{n^l,1} \\ 
		w^l_{1,2}, & w^l_{2,2}, & ..., & w^l_{n^l,2} \\
		\vdots & \vdots & \ddots & \vdots \\
		w^l_{1,n^{l-1}}, & w^l_{2,n^{l-1}}, & ..., & w^l_{n^l,n^{l-1}}
	}
\right]
\left[
	\matrix{
		\delta^{l, 1}_1, & \delta^{l, 2}_1, & ..., & \delta^{l, M}_1 \\
		\delta^{l, 1}_2, & \delta^{l, 2}_2, & ..., & \delta^{l, M}_2 \\
		\vdots & \vdots & \ddots & \vdots \\
		\delta^{l, 1}_{n^{l}}, & \delta^{l, 2}_{n^{l}}, & ..., & \delta^{l, M}_{n^{l}} \\
	}
\right],
$$
 which we can write more compactly as
$$
\boldsymbol{\Delta}^{l-1} = f'(\textbf{Z}^{l-1}) \odot \left( \textbf{W}^l \right)^T \boldsymbol{\Delta}^{l},
$$
which represents BP2.2.

### BP3.2

Remember that the real quantities of interest during backpropagation are the gradients of the *cost* function w.r.t. the weights and biases, because we need those to adjust the weights and biases into the direction so that the cost decreases. Also, recall that the cost is just the averaged loss over $M$ training examples, we know that 
$$
\frac{\partial C}{\partial \textbf{w}^l} = \frac{1}{M} \sum^M_{m=1} \frac{\partial L^m}{\partial \textbf{w}^l},
$$
where $\textbf{w}^l$ is the aforementioned flattened weight matrix in layer $l$​ and $L^m$ is the loss associated with the $m$-th training example. 

From BP3.1, we know that 
$$
\frac{\partial L}{\partial \textbf{w}^l} = 
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
		a^{l-1}_1, & a^{l-1}_2, & ..., & a^{l-1}_{n^{l-1}}
	}
\right],
$$
so, using that, we can rewrite equation (73) as follows
$$
\frac{\partial C}{\partial \textbf{w}^l} = 
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
		a^{l-1, m}_1, & a^{l-1, m}_2, & ..., & a^{l-1, m}_{n^{l-1}}
	}
\right].
$$
Working out the above matrix multiplication yields
$$
\frac{\partial C}{\partial \textbf{w}^l} = 
\frac{1}{M} \sum^M_{m=1} 
\left[
	\matrix{
		\delta^{l, m}_1 a^{l-1, m}_1, & \delta^{l, m}_1 a^{l-1, m}_2, & ..., & \delta^{l, m}_1 a^{l-1, m}_{n^{l-1}} \\
		\delta^{l, m}_2 a^{l-1, m}_1, & \delta^{l, m}_2 a^{l-1, m}_2, & ..., & \delta^{l, m}_2 a^{l-1, m}_{n^{l-1}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\delta^{l, m}_{n^l} a^{l-1, m}_1, & \delta^{l, m}_{n^l} a^{l-1, m}_2, & ..., & \delta^{l, m}_{n^l} a^{l-1, m}_{n^{l-1}}
	}
\right].
$$
Moving the summation inwards gives us 
$$
\frac{\partial C}{\partial \textbf{w}^l} = 
\frac{1}{M}
\left[
	\matrix{
		\sum^M_{m=1} \delta^{l, m}_1 a^{l-1, m}_1, & \sum^M_{m=1} \delta^{l, m}_1 a^{l-1, m}_2, & ..., & \sum^M_{m=1} \delta^{l, m}_1 a^{l-1, m}_{n^{l-1}} \\
		\sum^M_{m=1} \delta^{l, m}_2 a^{l-1, m}_1, & \sum^M_{m=1} \delta^{l, m}_2 a^{l-1, m}_2, & ..., & \sum^M_{m=1} \delta^{l, m}_2 a^{l-1, m}_{n^{l-1}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\sum^M_{m=1} \delta^{l, m}_{n^l} a^{l-1, m}_1, & \sum^M_{m=1} \delta^{l, m}_{n^l} a^{l-1, m}_2, & ..., & \sum^M_{m=1} \delta^{l, m}_{n^l} a^{l-1, m}_{n^{l-1}}
	}
\right].
$$
Notice that each cell contains a scalar product, which should ring a bell, because when multiplying two matrices with each other, each cell in the resulting matrix is a scalar product of a row of the left matrix and a column of the right matrix. Using this insight, we can decompose the above equation into the following matrix multiplication
$$
\frac{\partial C}{\partial \textbf{w}^l} = 
\frac{1}{M}
\left[
	\matrix{
		\delta^{l, 1}_1, & \delta^{l, 2}_1, & ..., & \delta^{l, M}_1 \\
        \delta^{l, 1}_2, & \delta^{l, 2}_2, & ..., & \delta^{l, M}_2 \\
        \vdots & \vdots & \ddots & \vdots \\
        \delta^{l, 1}_{n^l}, & \delta^{l, 2}_{n^l}, & ..., & \delta^{l, M}_{n^l}
	}
\right]
\left[
	\matrix{
		a^{l-1, 1}_1, & a^{l-1, 1}_2, & ..., & a^{l-1, 1}_{n^{l-1}} \\
		a^{l-1, 2}_1, & a^{l-1, 2}_2, & ..., & a^{l-1, 2}_{n^{l-1}} \\
		\vdots & \vdots & \ddots & \vdots \\
		a^{l-1, M}_1, & a^{l-1, M}_2, & ..., & a^{l-1, M}_{n^{l-1}}
	}
\right],
$$
or written more compactly as
$$
\frac{\partial C}{\partial \textbf{w}^l} = \frac{1}{M} \boldsymbol{\Delta}^l \left( \textbf{A}^{l-1} \right)^T,
$$
which represents equation BP3.2. 

### BP4.2

Finally, the other real quantity of interest is the gradient of the cost function w.r.t. the biases. Again, using the fact that the cost is an average over $M$​ training examples, we can deduce that
$$
\frac{\partial C}{\partial \textbf{b}^l} = \frac{1}{M} \sum^M_{m=1} \frac{\partial L^m}{\partial \textbf{b}^{l}}.
$$
Remember from BP4.1 that 
$$
\frac{\partial L}{\partial \textbf{b}^{l}} = \left[
	\matrix{
		\delta^l_1, & \delta^l_2, & ..., & \delta^l_{n^l}
	}
\right],
$$
which, first of all, we will redefine as its transpose to remain conform with the notations used from BP1.2 to BP3.2, i.e.
$$
\frac{\partial L}{\partial \textbf{b}^{l}} \coloneqq
\left( \frac{\partial L}{\partial \textbf{b}^{l}} \right)^T = 
\left[
	\matrix{
		\delta^l_1 \\ 
		\delta^l_2 \\
        \vdots \\
        \delta^l_{n^l}
	}
\right].
$$
From here on out, it is really straight forward. Plugging (82) back into (80) yields
$$
\frac{\partial C}{\partial \textbf{b}^l} = 
\frac{1}{M} \sum^M_{m=1}
\left[
	\matrix{
		\delta^{l, m}_1 \\ 
		\delta^{l, m}_2 \\
        \vdots \\
        \delta^{l, m}_{n^l}
	}
\right],
$$
or written more compactly as
$$
\frac{\partial C}{\partial \textbf{b}^l} = \frac{1}{M} \sum^M_{m=1} \boldsymbol{\delta}^{l, m},
$$
which represents equation BP4.2. 

### Summary

To summarize, in our backpropagation module, we want to implement the following 4 equations:

- BP1.2: $\boldsymbol{\Delta}^L = \frac{\partial L}{\partial \textbf{A}^L} \odot f'(\textbf{Z}^L)$ 
- BP2.2: $\boldsymbol{\Delta}^{l-1} = f'(\textbf{Z}^{l-1}) \odot \left( \textbf{W}^l \right)^T \boldsymbol{\Delta}^l$ 
- BP3.2: $\frac{\partial C}{\partial \textbf{w}^l} = \frac{1}{M} \boldsymbol{\Delta}^l \left( \textbf{A}^{l-1} \right)^T$ 
- BP4.2: $\frac{\partial C}{\partial \textbf{b}^l} = \frac{1}{M} \sum^M_{m=1} \boldsymbol{\delta}^{l, m}$

## Why Backpropagation?

You might wonder why we should bother trying to derive a complicated algorithm and not use other seemingly simpler methods for computing all partial derivatives in the network. To motivate the need for the backpropagation algorithm, assume we simply wanted to compute the partial derivative of weight $w_j$ as follows
$$
\frac{\partial{L}}{\partial{w_j}} = \frac{L(\textbf{w} + \epsilon \textbf{e}_j, \textbf{b}) - L(\textbf{w}, \textbf{b})}{\epsilon},
$$
where $\textbf{w}$​​ and $\textbf{b}$​​ are flattened vectors containing all weights and biases of the network, where $\epsilon$​​ is a infinitesimal scalar and where $\textbf{e}_j$​​ is the unit vector being $1$​​ at position $j$​​ and $0$​​ elsewhere. Assuming that our network has one million parameters, we would need to calculate $L(\textbf{w} + \epsilon \textbf{e}_j, \textbf{b})$​ a million times (once for each $j$​), and also, we would need to calculate $L(\textbf{w}, \textbf{b})$​ once, summing up to a total of $1,000,001$​ forward passes for just a *single* training example! As we will see in this section, the backpropagation algorithm let's us compute all partial derivatives of the network with just one forward- and one backward pass through the network!

# Gradient Descent

In the previous section, we described how to compute the gradients, which mathematically speaking, point into the *direction* of the steepest ascent of the cost function. In this section, we will describe how to use the gradients in order to *update* the weights and biases such that the cost decreases. 

We will use a very simple way of updating the weights and biases which is called *Stochastic Gradient Descent* (SGD). Assuming that we have calculated BP3.2 and BP4.2, we can perform the weight updates as
$$
\textbf{w}^{l}_{s} = \textbf{w}^{l}_{s-1} - \lambda \left( \frac{\partial C}{\partial \textbf{w}^l} \right)_{s-1},
$$
and similarly, the bias updates as
$$
\textbf{b}^{l}_s = \textbf{b}^{l}_{s-1} - \lambda \left( \frac{\partial C}{\partial \textbf{b}^l} \right)_{s-1},
$$
for update steps $i = 1, 2, ..., S$. Notice that $\textbf{w}^{l}_{s=0}$ and $\textbf{b}^{l}_{s=0}$ are initialized randomly, $S$ represents the number of update steps and $\lambda$ represents the *learning rate* controlling the step size toward the local (and hopefully global) minimum of the cost function. 

Assuming that we divided our dataset into batches with $M$ training examples each, we will end up with $S$ batches, where $S$ is computed as 
$$
S = \text{round\_up}(N/M),
$$
where $\text{round\_up}$​ is a function that always rounds a floating point number *up* to the nearest integer and where $N$ represents the number of all training examples in total. Notice that $S$ always needs to be rounded up in order to make sure that during one *epoch*[^epoch], all training examples have been forward- and backward propagated through the network. 

[^epoch]: During one epoch, all training examples have been forward- and backward propagated through the network. Usually, neural networks will need many (50-100) of such epochs to accurately predict the target values. Notice, that during each epoch, $S$​ gradient descent update steps are performed.  

# Activation and Loss Functions

In the previous sections, we used some specific loss and activation functions and in this section we want to show some other common choices. 

## Activation Functions

### Sigmoid

We have already taken a sneak peek at the sigmoid function during the forward propagation, so that the reader has a general idea what we are talking about, but now, we will delve into more depth. The sigmoid function can be expressed as
$$
f(z) = \frac{1}{1 + e^{-z}}.
$$

Notice that for very large positive inputs, $f(z) \rightarrow 1$​ and that for very large negative inputs, $f(z) \rightarrow 0$ and that $f(z) = 0.5$ if $z=0$. 

The corresponding derivative is
$$
f'(z) = f(z) \times (1 - f(z)),
$$
from which we can see that for very large inputs (positive as well as negative), $f'(z) \rightarrow 0$ and that $f'(z) = 0.25$ if $z=0$​​​. The fact that the sigmoid function's derivative is close to 0 for very large inputs causes the errors (at each layer where the sigmoid activation function is used) to be very small, which in turn causes the gradients to contain very small values. This problem is called *learning slowdown* (briefly mentioned earlier) and when calculating the error at the output layer, this problem can be solved by using the categorical cross entropy cost function as we have shown in the examples for equations BP1.1 and BP1.2. 

### ReLU

The Rectified Linear Unit (ReLU) is defined as
$$
f(z) = max(0, z),
$$
Its derivative is actually not defined for $z=0$​, but in practice, the probability that $z=0$​ exactly is infinitesimal. So, we will define the derivative of the ReLU function as 
$$
f'(z) = 
\begin{cases}
z & \text{if} & z > 0 \\
0 & \text{if} & z \leq 0
\end{cases}
$$
Notice from equation (31), that if e.g. $f'(z^L_1) = 0$​​, $\delta^L_1 = 0$​​. Then, by looking at equation (57), we know that 
$$
\begin{array}{c}
\delta^L_1 a^{L-1}_1 = 0, \\
\delta^L_1 a^{L-1}_2 = 0, \\
\vdots \\
\delta^L_1 a^{L-1}_{n^{l-1}} = 0,
\end{array}
$$


which in turn means that 
$$
\frac{\partial L}{\partial \textbf{w}^L}[1, 2, ..., n^{l-1}] = 0,
$$
namely that elements $1$​ to $n^{l-1}$​ of the flattened weight gradient are zero. Simply put, this means that for a particular training example, all weights connected to neuron $1$​ in layer $L$​​​ won't get updated if $f'(z^L_1) = 0$. 

### tanh

TODO

### Softmax

Unlike all other activation functions discussed so far, the softmax function takes a vector as input and outputs a vector whose elements sum up to $1$. It is defined as follows
$$
f(\textbf{z}^l) = 
\left[
	\matrix{
		\frac{e^{z_1}}{\sum^{n^l}_{j=1} e^{z_j}}, & \frac{e^{z_2}}{\sum^{n^l}_{j=1} e^{z_j}}, & ..., & \frac{e^{z_{n^l}}}{\sum^{n^l}_{j=1} e^{z_j}}
	}
\right]
$$


## Loss functions

TODO

### Categorical Crossentropy

TODO

### Mean Squared Error

TODO

# Optimization Methods

TODO

## Weight Initialization

TODO
