# Neural Networks from Scratch

This repository is intended for anyone who still considers himself a deep learning student and who wants to understand the nitty-gritty details of the equations behind the back-propagation algorithm and how to implement them with [NumPy](https://numpy.org/). This repository is not intended for anyone who is not interested in what is going on under the hood of neural networks. 

Currently, this repository is basically a self-implemented version of the [Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) of the [Tensorflow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras) deep learning library, i.e. it supports network architectures which are a stack of layers chained behind each other. More complex architectures including e.g. residual connections are not supported. At the moment, only the following building blocks are supported, but may easily be extended:

- Dense layers
- Categorical Cross-Entropy losses
- Image data generators
- Stochastic Gradient Descent optimizers

# Acknowledgments 

The knowledge used in this repository stems from a variety of sources. The below is probably an incomplete list of helpful sources, but it is definitely a good start.

- [Andrew Ng's Deep Learning Courses on Coursera](https://www.coursera.org/specializations/deep-learning?utm_source=gg&utm_medium=sem&utm_campaign=17-DeepLearning-ROW&utm_content=17-DeepLearning-ROW&campaignid=6465471773&adgroupid=77415260637&device=c&keyword=coursera%20deep%20learning%20ai&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=506751438660&hide_mobile_promo&gclid=CjwKCAjw-sqKBhBjEiwAVaQ9ayqogdXIcEIxKgM1lXbJaUr4DgI5nEdHSjA9pp8Q2b3x8nFMgVo80BoCusIQAvD_BwE)
- [Micheal Nielsons online Deep Learning book](http://neuralnetworksanddeeplearning.com/)
- [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown)

 I also want to thank my professor and mentor Prof. Dr. Knoblauch for valuable feedback and detailed study material. 

# Theory

The theory behind Multi-Layer-Perceptrons (Feed Forward Neural Networks consisting of a stack of dense layers) can be found in `theory/MLP_theory.html`. To open this document, you will need to download the whole repository first though. The easiest way to do that is to just download it as a zip file by clicking on the green [Code](https://github.com/kevinsuedmersen/neural_networks_from_scratch/archive/refs/heads/master.zip) button or by executing `git clone https://github.com/kevinsuedmersen/neural_networks_from_scratch.git` from the command line.  

# Example Usage

## MNIST Digit Classification

Download data from [kaggle](https://www.kaggle.com/jidhumohan/mnist-png) and unzip it into the `resources/mnist_png` directory. Then, simply execute the code inside the `mnist_digits_classification_example.py` file inside the project root directory.  With the configurations used, the results on the test set are:

| Categorical Cross-Entropy | Accuracy | Precision | Recall |
| ------------------------- | -------- | --------- | ------ |
| 0.166                     | 0.990    | 0.951     | 0.951  |

The packages used in `mnis_digits_classification_example.py` stem from the `src/lib`. All other files are probably irrelevant for you at this point and only exist so that this repository can be easily extended for other deep learning use cases later. 

# Feedback

Let's make this repository better together, so if you have some feedback, feel free to create an issue, fork this repository, or ideally, become a contributor!

