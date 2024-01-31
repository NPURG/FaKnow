SAFE
====
Introduction
-------------
`[paper] <https://dl.acm.org/doi/abs/10.1007/978-3-030-47436-2_27>`_

**Title:** SAFE: Similarity-Aware Multi-Modal Fake News Detection

**Authors:** Xinyi Zhou, Jindi Wu, Reza Zafarani

**Abstract:** Effective detection of fake news has recently attracted significant attention. Current studies have made
significant contributions to predicting fake news with less focus on exploiting the relationship (similarity) between
the textual and visual information in news articles. Attaching importance to such similarity helps identify fake news
stories that, for example, attempt to use irrelevant images to attract readers’ attention. In this work, we propose a
imilarity-ware ak news detection method () which investigates multi-modal (textual and visual) information of news
articles. First, neural networks are adopted to separately extract textual and visual features for news representation.
We further investigate the relationship between the extracted features across modalities. Such representations of news
textual and visual information along with their relationship are jointly learned and used to predict fake news.
The proposed method facilitates recognizing the falsity of news articles based on their text, images, or their “mismatches.”
We conduct extensive experiments on large-scale real-world data, which demonstrate the effectiveness of the proposed method.

.. image:: ../../../media/SAFE.png
    :align: center

For source code, please refer to :ref:`SAFE <faknow.model.content_based.multi_modal.safe>`

If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../../user_guide/config_intro`
- :doc:`../../../../user_guide/data_intro`
- :doc:`../../../../user_guide/train_eval_intro`
- :doc:`../../../../user_guide/usage`
