Customize Evaluator
=================
The :ref:`Evaluator <faknow.evaluate>` class is used to evaluate the performance of a model using different evaluation metrics. The class is
initialized to accept a list of metrics, where each metric can be a string representing a built-in metric function
(e.g., accuracy, precision, recall, f1, auc), or a custom Callable function. If no metrics are provided, accuracy,
precision, recall, and f1 will be used by default, and if you want to insert your own custom metrics function into
``Evaluator``, you can do so by creating a Callable function that follows the signature of metric_func(outputs: Tensor,
y: Tensor) -> float. Tensor) -> float signature, which is then passed to the ``Evaluator`` constructor as part of the metrics list.