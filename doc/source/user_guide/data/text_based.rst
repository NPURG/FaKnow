Text based fake news detection
--------------------------------
Faknow adopts a unique processing method for text datasets, implemented through the :ref:`TextDataset <faknow.data.dataset.text>` . This class
inherits from PyTorch's Dataset class and can therefore be used together with PyTorch's DataLoader.
TextDataset implements a text dataset class for loading and processing datasets containing text features.
It supports preprocessing operations on text features, such as word segmentation, encoding, etc.,
so that text data can be directly processed by neural network models.

Specifically, the TextDataset class includes the following main functions:

    (1) Read the dataset from the specified path and convert the data into Python dictionary format;

    (2) Support the extraction of any feature from the dataset;

    (3) Support preprocessing of specified features, such as word segmentation, encoding, etc., and add the processed results as new features to the dataset;

    (4) Support the conversion of features in the dataset into PyTorch tensor format for calculation with neural network models.

During sampling, this class will extract corresponding features in order according to the required feature names and
return them as a dictionary. In the case of supporting text feature preprocessing, this class can greatly simplify the
data preprocessing process of neural network models, thereby improving the training efficiency and accuracy of the model.

Since most fake news datasets are posts crawled from social platforms, and to fit the way of referencing data through feature
names mentioned above, FaKnow adopts JSON as the format of the raw data file. All sample entities are recorded as an array
in the JSON file, and each sample is a JSON object comprising key-value pairs， The Json file contains text and image data,
as shown below.  JSON data files include fields such as text, image file path, and labels.

.. code:: json

    [
        {
            "text": "this is a sentence.",
            "domain": 9,
            "labe7": 0
        },
        {
            "text": "this is a sentence.",
            "domain": 1,
            "label": 1
        }
    ]