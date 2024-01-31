Customize Models
=================
Here, we present how to develop a new model, and apply it to the Faknow.

Faknow supports content based, social context, knowledge aware fake news detection algorithms.

In order to implement a new false information recognition algorithm, we can use the API provided by Faknow to implement
the algorithm. In this process, we will use three functions to construct a new model.

Implement __init__()
------------------------
We need to implement the ``init()`` function for parameter initialization and global variable definition. This function is a
subclass of the abstract model class ``AbstractModel`` provided in our library, which is used to instantiate our new model.

Implement calculate_loss()
------------------------
We need to implement ``calculate_loss()`` function that calculates the loss of the new model to optimize the training effectiveness of the
model. Based on the return value of this function, the library will call the optimization method and train the model
according to the preset configuration. When implementing this function, we need to consider selecting appropriate loss
functions and optimizers to improve the accuracy and efficiency of the model.

Implement predict()
------------------------
We need to implement the ``predict()`` function, which is used to predict the category and probability of false information.
This function will return a binary containing the predicted probabilities of real and false categories. When implementing
this function, we need to consider selecting appropriate classifiers and prediction methods to improve the accuracy and
efficiency of the model.

Specifically, we implement the ``init()``, ``calculate_loss()``, ``predict()`` function can create an
accurate and efficient false information recognition model. In short, using the API provided by Faknow to implement a
new false information recognition algorithm model can make our code more concise, unified, and efficient.