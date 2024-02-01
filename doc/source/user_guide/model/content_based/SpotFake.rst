SpotFake
========
Introduction
-------------
`[paper] <https://ieeexplore.ieee.org/document/8919302>`_

**Title:** SpotFake: A Multi-modal Framework for Fake News Detection

**Authors:** Shivangi Singhal, Rajiv Ratn Shah, Tanmoy Chakraborty, Ponnurangam Kumaraguru, Shin'ichi Satoh

**Abstract:** A rapid growth in the amount of fake news on social media is a very serious concern in our society. It is
usually created by manipulating images, text, audio, and videos. This indicates that there is a need of multimodal system
for fake news detection. Though, there are multimodal fake news detection systems but they tend to solve the problem of
fake news by considering an additional sub-task like event discriminator and finding correlations across the modalities.
The results of fake news detection are heavily dependent on the subtask and in absence of subtask training, the performance
of fake news detection degrade by 10% on an average. To solve this issue, we introduce SpotFake-a multi-modal framework
for fake news detection. Our proposed solution detects fake news without taking into account any other subtasks.
It exploits both the textual and visual features of an article. Specifically, we made use of language models (like BERT)
to learn text features, and image features are learned from VGG-19 pre-trained on ImageNet dataset. All the experiments
are performed on two publicly available datasets, i.e., Twitter and Weibo. The proposed model performs better than the
current state-of-the-art on Twitter and Weibo datasets by 3.27% and 6.83%, respectively.

.. image:: ../../../media/SpotFake.png
    :align: center

For source code, please refer to :ref:`SpotFake <faknow.model.content_based.multi_modal.spotfake>`

If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../../user_guide/config_intro`
- :doc:`../../../../user_guide/data_intro`
- :doc:`../../../../user_guide/train_eval_intro`
- :doc:`../../../../user_guide/usage`