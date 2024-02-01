Customize Trainers
=================
FaKnow designed a basic trainer, :ref:`BaseTrainer <faknow.train.trainer>`, in the training module. This trainer implements the basic functions of
model training, with the specified ``model``, ``loss_func``, ``optimizer``, ``evaluator`` and ``scheduler``, the algorithm model is trained,
and each epoch validates the model. The process is presented in various visual ways, and the trained algorithm model is
finally saved.

If the training of the algorithm only requires the most basic work, simply call BaseTrainer; If the training of
algorithms requires additional work, Faknow has written specific trainers for these specific algorithms which inherit
BaseTrainer, For example, ``MFANTrainer`` is a trainer written specifically for MFAN, which minimizes code redundancy and
improves code readability.