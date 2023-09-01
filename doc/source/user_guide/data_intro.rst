Data Module Introduction
=========================
In the data module, Faknow receives the original dataset transmitted by the user and processes it according to the
situation to the data format required for model input. According to the characteristics of news information,
there are two types of dataset preprocessing methods: text dataset and multimodal dataset, which are respectively
suitable for text based false news detection and multimodal information based false news detection.

Here are two ways to preprocess datasets:

.. toctree::
   :maxdepth: 1

   data_flow
   text_based
   multi_info_based