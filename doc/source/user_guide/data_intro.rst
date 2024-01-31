Data Module Introduction
=========================
In the data module, Faknow receives the original dataset transmitted by the user and processes it according to the
situation to the data format required for model input. According to the characteristics of news information,
there are two types of dataset preprocessing methods: ``text dataset`` and ``multimodal dataset``, which are respectively
suitable for text based fake news detection and multimodal information based fake news detection.

Here are two ways to preprocess datasets:

.. toctree::
   :maxdepth: 2

   data/data_flow
   data/text_based
   data/multi_info_based