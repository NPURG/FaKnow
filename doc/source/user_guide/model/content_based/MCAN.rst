MCAN
====
Introduction
-------------
`[paper] <https://aclanthology.org/2021.findings-acl.226/>`_

**Title:** Multimodal Fusion with Co-Attention Networks for Fake News Detection

**Authors:** Yang Wu, Pengwei Zhan, Yunjian Zhang, Liming Wang, Zhen Xu

**Abstract:** Fake news with textual and visual contents has a better story-telling ability than text-only contents, and
can be spread quickly with social media. People can be easily deceived by such fake news, and traditional expert identification
is labor-intensive. Therefore, automatic detection of multimodal fake news has become a new hot-spot issue. A shortcoming
of existing approaches is their inability to fuse multimodality features effectively. They simply concatenate unimodal
features without considering inter-modality relations. Inspired by the way people read news with image and text, we propose
a novel Multimodal Co-Attention Networks (MCAN) to better fuse textual and visual features for fake news detection.
Extensive experiments conducted on two realworld datasets demonstrate that MCAN can learn inter-dependencies among multimodal
features and outperforms state-of-the-art methods.


.. image:: ../../../media/MCAN.png
    :align: center

Running with Faknow
---------------------
**Model Hyper-Parameters:**

- ``bert (str)`` : bert model, default = ``'bert-base-chinese'``

- ``max_len (int)`` : max length of text, default = ``255``

- ``batch_size (int)`` : batch size, default = ``16``

- ``num_epochs (int)`` : number of epochs, default = ``100``

- ``metrics (List)`` : metrics, if None, ['accuracy', 'precision', 'recall', 'f1'] will be used, default = ``None``

- ``device (str)`` : device, default = ``'cpu'``

- ``**optimizer_kargs`` : optimizer kargs

**A Running Example:**

Write the following code to a python file, such as run.py

.. code:: python

    from faknow.run.content_based.multimodal import run_mcan

    run_mcan(train_path=, validate_path=, test_path=)

And then:

.. code:: bash

   python run.py

If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../../user_guide/config_intro`
- :doc:`../../../../user_guide/data_intro`
- :doc:`../../../../user_guide/train_eval_intro`
- :doc:`../../../../user_guide/usage`