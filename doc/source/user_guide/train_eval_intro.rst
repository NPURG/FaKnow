Training & Evaluation Introduction
==================================

Training introduction
---------------------
Faknow designed a basic trainer,
`BaseTrainer <http://127.0.0.1:8000/faknow/faknow.train.html#faknow.train.trainer.BaseTrainer>`_, in the training module.
This trainer implements the basic functions of
model training, with the specified ``model``, ``loss_func``, ``optimizer``, ``evaluator`` and ``scheduler``,
the algorithm model is trained, and each epoch validates the model. The process is presented in various visual ways,
and the trained algorithm model is finally saved.

If the training of the algorithm only requires the most basic work, simply call BaseTrainer; If the training of
algorithms requires additional work, Faknow has written specific trainers for these specific algorithms,
For example, `MFANTrainer <http://127.0.0.1:8000/faknow/faknow.train.html#faknow.train.pgd_trainer.MFANTrainer>`_
is a trainer written specifically for MFAN, which minimizes code redundancy and improves code readability.

Evaluation introduction
------------------------
Faknow has designed a validator Evaluator in the validation module, which uses the input model output tensor ``outputs``
and the real label tensor ``y`` to evaluate the model's indicators such as ``accuracy``, ``precision``, ``recall``, and ``f1``.