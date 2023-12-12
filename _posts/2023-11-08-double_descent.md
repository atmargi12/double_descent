---
layout: distill
title: Dynamic Ensemble Learning for Mitigating Double Descent
description: Exploring when and why Double Descent occurs, and how to mitigate it through Ensemble Learning.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Mohit Dighamber
    affiliations:
      name: MIT
  - name: Andrei Marginean
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-double_descent.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Motivation
  - name: Related Work
  - name: Methods
    subsections:
    - name: Decision Trees
    - name: Random Forest
    - name: Logistic Regression
    - name: Support Vector Machines
    - name: Neural Networks
  - name: Evaluation
    subsections:
    - name: Software
    - name: Datasets
    - name: Computing Resources
    - name: Reproducibility Statement

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---


## Abstract

We outline the fundamental 'bias-variance tradeoff' concept in machine learning, as well as how the double descent phenomenon counterintuitively bucks this trend for models with levels of parameterization at or beyond the number of data points in a training set. We present a novel investigtaion of the mitigation of the double descent phenomenon by coupling overparameterized neural networks with each other as well as various weak learners. Our findings demonstrate that coupling neural models results in decreased loss during the variance-induced jump in loss before the interpolation threshold, as well as a considerable improvement in model performance well past this threshold. Machine learning practitioners may also find useful the additional dimension of parallelization allowed through ensemble training when invoking double descent. 

## Motivation

There are many important considerations that machine learning scientists and engineers
must consider when developing a model. How long should I train a model for? What features
and data should I focus on? What exactly is an appropriate model size? This last question
is a particularly interesting one, as there is a bit of contention regarding the correct answer
between different schools of thought. A classical statistician may argue that, at a certain
point, larger models begin to hurt our ability to generalize. By adding more and more
parameters, we may end up overfitting to the training data, resulting in a model that poorly
generalizing on new samples. On the other hand, a modern machine learning scientist may
contest that a bigger model is always better. If the true function relating an input and output
is conveyed by a simple function, In reality, neither of these ideas are completely correct in
practice, and empirical findings demonstrate some combination of these philosophies.
This brings us to the concept known as **double descent**. Double descent is the phenomenon
where, as a model’s size is increased, test loss increases after reaching a minimum, then
eventually decreases again, potentially to a new global minimum. This often happens in the
region where training loss becomes zero (or whatever the ’perfect’ loss score may be), which
can be interpreted as the model ’memorizing’ the training data given to it. Miraculously,
however, the model is not only memorizing the training data, but learning to generalize as
well, as is indicated by the decreasing test loss.

The question of ’how big should my model be?’ is key to the studies of machine learning
practitioners. While many over-parameterized models can achieve lower test losses than the
initial test loss minimum, it is fair to ask if the additional time, computing resources, and
electricity used make the additional performance worth it. To study this question in a novel
way, we propose incorporating **ensemble learning**.

Ensemble learning is the practice of using several machine learning models in conjunction
to potentially achieve even greater accuracy on test datasets than any of the individual
models. Ensemble learning is quite popular for classification tasks due to this reduced error
empirically found on many datasets. To our knowledge, there is not much literature on how
double descent is affected by ensemble learning versus how the phenomenon arises for any
individual model.

We are effectively studying two different **types** of model complexity: one that incorporates
higher levels parameterization for an individual model, and one that uses several models in
conjunction with each other. We demonstrate how ensemble learning affects the onset of the
double descent phenomenon. By creating an ensemble that includes an overparameterized
neural network, which can take extreme amounts of time and resources to generate, with
overparameterized machine learning models, we will show the changes in the loss curve,
specifically noting the changes in the regions where double descent is invoked. We hope that the results we have found can potentially be used by machine learning researchers and engineers to
build more effective models.

***

## Related Work

One of the first papers discussing double descent was ’Reconciling modern machine-
learning practice and the classical bias–variance trade-off’ by Belkin et al. <d-cite key="belkin2019reconciling"></d-cite>. This paper
challenged the traditional idea of the ’bias-variance tradeoff’. The bias-variance tradeoff is
a fundamental concept in machine learning that describes the tension between two types of
model error: bias and variance. Bias is the error between the expected prediction of the
model and the true output value, introduced by approximating a real-world quantity with
a model, which may overisimplify the true problem at hand. Variance refers to the error
due to a model’s sensitivity to small fluctuations in the training dataset. Overfitted models
may have high variance, as they may model random noise in the data as well. In short,
classical statistical learning argues that there is some optimal level of parametrization of
a model, where it is neither underparameterized or overparameterized, that minimizes the
total error between bias and variance. However, Belkin’s paper finds that, empirically, this
bias-variance tradeoff no longer becomes a tradeoff at a certain level of overparamateriza-
tion. They showed that after the interpolation threshold (where the model fits perfectly to
the training data), test error eventually began to decrease once again, even going below the
error deemed optimal by the bias-variance minimum.



Nakkiran et al.’s ’Deep Double Descent: Where Bigger Models and More Data Hurt’ <d-cite key="nakkiran2021deep"></d-cite> expanded these findings to the realm of **deep** learning. In this work, double descent is shown to occur for both large models and large datasets. Additionally, this paper demonstrates that,
counterintuitively, adding more data at a certain point actually worsened the performance
of sufficiently large models. Specifically, this occurred at and close to the interpolation
threshold for neural models. For the region between the first and second loss minima, model
performance can suffer greatly, despite the increased computational time and resources used
to generate such models. While this region of the test loss curve is typically not a level of
parameterization that one would use in practice, understanding this the entire loss curve
can help practitioners for several reasons. For one, this degraded phase of performance can
be crucial for tweaking model architecture and adjusting training strategies. This is key to
discovering if one’s model is robust and adaptable to various other datasets and tasks. This
highlights the need for a new understanding for model selection for effectively generalizing
to testing datasets better that can mitigate decreases in model performance and invoke a
second loss minimum quickly.

In the classic paper ’Bagging Predictors’, Breiman describes the concept of combining the
decisions of multiple models to improve classification ability <d-cite key="breiman1996bagging"></d-cite>. Empirically, this bootstrap aggregating, or ’bagging’ technique, reduced variance and improved accuracy, outperforming the single predictors that comprised the ensemble model. We present a novel combination
of the findings of this paper with the double descent phenomenon. Effectively, by increasing model complexity via overparameterization and ensemble learning, we aim to study if this combination can mitigate loss increases and invoke a second loss minimum with smaller models.

***

## Methods

### Computing Resources and Software

We have implemented this project using CUDA and the free version of Google Colab. To train and test these models, we use various machine learning packages in Python, namely Scikit-learn, PyTorch and Tensorflow. Additional software commonly used for machine learning project such as numpy and matplotlib was also utilized.

### Data

We use the MNIST dataset for this report <d-cite key="deng2012mnist"></d-cite>. MNIST is a popular dataset used for image classification, where each sample image is a 28 by 28 grayscale image of a written integer between 0 and 9, inclusive. Each image comes with the true label of the image's integer. This data is publicly available for experimentation, and our use of it does not pose any ethical or copyright concerns. 

For this project, we use the MNIST dataset to unearth the double descent phenomenon. At the moment, we intend to experiment with five models, as well as an ensemble of them: decision trees, random forest, logistic regression, support vector machines, and small neural networks. We choose these models because of their ability to be used for classification tasks, and more complicated models run the risk of exceeding Google Colab's limitations, especially when we overparameterize these models to invoke double descent. 

### Decision Trees

Decision trees are a machine learning model used for classification tasks. This model resembles a tree, splitting the data at branches, culminating in a prediction at the leaves of the tree. 

To invoke double descent for decision trees, we can start with a small maximum depth of our tree, and increase this parameter until the training loss becomes perfect. Note that by increasing the max depth by 1, the maximum number of leaves doubles, indicating this model has exponential growth in its complexity with respect to the tree depth.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/dd_decision_tree_zero_one_8.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### AdaBoost Tree

Adaptive Boosting (AdaBoost) itself is an ensemble model used for robust classification. Freund et al.'s paper 'A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting' first introduced the algorithm <d-cite key="freund1997decision"></d-cite>. On a high level, this paper describes how boosting is especially effective when sequentially combining weak learners that are moderately inaccurate (in this case, these are decision trees) to create a strong learner. We study the loss curve of the AdaBoost model as we increase the number of estimators, that being the number of trees.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/dd_adaboost_zero_one_2.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


### L2-Boost Tree

L2 Boosting is quite similar to the AdaBoost model, except for L2 Boosting, as models are built sequentially, each new model in the boosting algorithm aims to minimize the L2 loss. Like before, we will study the loss curve of the L2-Boosted model as we increase the number of estimators, or the number of trees. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/dd_l2boost_zero_one_1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


### Random Forest

Random forest is another model that is already an ensemble. As the name implies, this model is a collection of decision trees with randomly selected features, and like the singular decision tree, this model is used for classification tasks. We can begin random forest with a small number of trees, and increase this until we see the double descent phenomenon in our test loss.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/dd_rf_zero_one_6.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Logistic Regression

Logistic regression is a classic model used for estimating the probability a sample belongs to various classes. We induce overfitting in logistic regression by varying the ratio of the number of features over the amount of data. We gradually reduce this ratio similar to the methodology of Deng et. al in order to induce overfitting <d-cite key="logistic"></d-cite>.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/dd_logistic_regression_zero_one_c.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/dd_logistic_regression_zero_one_d.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


### Neural Networks

We use a neural network as our main model for the ensemble. Our deep learning model is a relatively small one, with variable width in one of the included layers. By increasing this width, we achieve perfect training loss.  

We define the general architecture of the neural network used in this report as follows: 

#### Network Layer

Let the input data be an $m$ by $m$ pixel image from the MNIST dataset, which can be processed as an $m$ by $m$ matrix, where entry $(i,j)$ is an integer between 0 and 255 (inclusive) representing the grayscale color of the pixel. Note that $m=28$ for MNIST, though for generality, we use $ m $ in this network definition. A value of 0 represents a black pixel, 255 is a white pixel, and values between these are varying shades of gray. We first flatten this structure into a $ m^2 $ by 1 vector, such that the entry $ (i,j) $ of the matrix becomes the $ j + 28*i$-th entry of the vector, using zero-indexing. We use this vector as the input of our neural network. 

Set $ n $ as the network width, which in our project will be varied in different tests. Let $ W^1 $ be an $ m \times n$  matrix, where $ W^1_{ij}$ is the weight of input $i$ applied to node $j$, and let $W^1_0$ be an $n \times 1$ column vector representing the biases added to the weighted input. For an input $X$, we define the **pre-activation** to be an $n \times 1$ vector represented by $Z = {W^1}^T X + W^1_0$. 

We then pass this linearly transformed vector to the ReLU activation function, defined such that 

$$
\begin{equation*}
\text{ReLU}(x)=\begin{cases}
          x \quad &\text{if} \, x > 0 \\
          0 \quad &\text{if} \, x \leq 0 \\
     \end{cases}
\end{equation*}
$$

We use this choice of activation function due to the well-known theorem of universal approximation. This theorem states that a feedforward network with at least one single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of $ \mathbb{R}^{m^2} $ if the ReLU activation function is used <d-cite key="hornik1991approximation"></d-cite>. Applying an activation function ReLU to each element of $Z $, the layer finally outputs 

$$
A = ReLU(Z) = ReLU(W^T X + W_0)
$$

Next, we will input $A$ into a second hidden layer of the neural network. Let $k$ be the number of classes that the data can possibly belong to. Again, $k = 10$ for MNIST, though we will use $k$ for generality. Then let $W^2$ be an $n$ by $k$ matrix, where $W^2_{ij}$ is the weight of input $i$ applied to node $j$, and let $W^2_0$ be a $k \times 1$ column vector representing the biases added to the weighted input. For input $A$, define a second pre-activation to be a $k \times 1$ vector represented by $B = {W^2}^T A + W^2_0$.

Finally, we pass $B$ to the softmax activation function, defined as

$$
softmax(z)_i = \frac{e^{z_i}}{\sum_{i=1}^{k} e^{z_j}}
$$

This will yield a $k \times 1$ vector of normalized probabilities of the input image belonging to any of the $k$ classes.

#### Training

Let class $i $ be the true classification for a data point. We have that $y_i = 1$, and for all $j \neq i$, $y_j = 0$. Furthermore, let $\hat{y_i}$ be the generated probability that the sample belongs to class $i$. The categorical cross-entropy loss is then defined as follows: 

$$
\mathcal{L}_{CCE} (y_i, \hat{y_i}) = - \sum_{i=0}^{9} y_i \log (\hat{y_i})
$$

From this computed loss, we use backpropagation and gradient descent with learning rate $\eta$ to optimize model weights. We run experiments that train over 100, 500, and 2000 epochs.

### Boostrap Aggregating

We use bootstrap aggregating, or 'bagging', to formulate our ensemble of these six models. Effectively, each model is given a certain number of 'votes' on what that model believes is the correct classification for any given MNIST sample image. The classification with the most total votes is then used as the ensemble's overall output. In the event of a tie, the neural network's prediction is chosen. Since we want a neural model to be the basis of our ensemble, we vary the number of votes assigned to the neural network while keeping the number of votes for other models fixed to 1. With five supplementary models in addition to the neural network, giving the neural network 5 or more votes is not necessary, since this ensemble would always output the same results as the neural network. Because of this, we study the loss curve when giving the neural network 1, 2, 3, and 4 votes. Note that decimal value votes for the neural network are not sensible, since it can be proved that all potential voting scenarios are encapsulated into the four voting levels we have chosen.

### Reproducibility Statement

To ensure reproducibility, we have included the codebase used for this project, as well as the above description of our data, models, and methods. 

***

## Results





***

## Discussion




One notable advantage to this ensemble method is the ability to further parallelize one's training of overparameterized neural networks. These models can take extreme lengths of time to train, and besides increasing the computational allocation used, practitioners may use data, model, or processor parallelism in order to reduce this time. The ensemble neural networks we use are independently generated, meaning that they can be trained on different machines without issue. This could be a valid alternative to training for more epochs for reducing model error past the interpolation threshold. 


***

## Conclusion



Ensembles consisting solely of neural networks resulted in a considerable boost in performance past the interpolation threshold. However, pairing the neural network with weak learners in an ensemble voting system actually **decreased** testing performance, though this adverse effect decreased as the neural network received proportionally more votes. Machine learning engineers that intend to intentionally overparameterize their models may take advantage of not only the ensemble approach's increased performance, but the enhanced parallelization capabilities offered by the method.

***

## Future Work


This project was implemented using Google Colab, which proved to be restrictive for adopting more complex models. A key part of the double descent phenomenon is overparameterization, and so complex models that are additionally overparameterized will require more powerful computing resources beyond what we used. Furthermore, additional computing power can allow for this project to be expanded to more complicated datasets and tasks. MNIST classification is computationally inexpensive, though invoking double descent in more complex tasks such as text generation in natural language processing was not feasible using Google Colab. Future projects that follow this work should keep computational limitations in mind when choosing models and datasets. 

During the planning process of this project, we discussed using a more rigorous voting system than what is traditionally found in ensemble model projects. Effectively, each model would have a weight associated with how much influence its output should have on the overall ensemble output. For $n$ models, each model could start with, say, a weight of $1/n$. Then, after producing each model's vector output, the categorical cross-entropy loss with respect the true output could be computed, and the weights of each model could be updated such that each model has its weight decreased by some amount proportional to the calculated loss. Then, these weights could be normalized using the softmax function. This would be repeated for each level of parameterization. Due to resource constraints, learning both the model weights and ensemble weights at each level of ensemble parameterization was not feasible given the size of the models we built, as well as the number of epochs we trained over. Future studies may wish to implement this method, however, to produce a more robust ensemble for classification.