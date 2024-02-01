TEXTCNN
=======
Introduction
-------------
`[paper] <https://aclanthology.org/D14-1181/>`_

**Title:** Convolutional Neural Networks for Sentence Classification

**Authors:** Yoon Kim

**Abstract:** We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained
word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static
vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further
gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both
task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks,
which include sentiment analysis and question classification.

.. image:: ../../../media/TEXTCNN.png
    :align: center

For source code, please refer to :ref:`TEXTCNN <faknow.model.content_based.textcnn>`

If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../../user_guide/config_intro`
- :doc:`../../../../user_guide/data_intro`
- :doc:`../../../../user_guide/train_eval_intro`
- :doc:`../../../../user_guide/usage`