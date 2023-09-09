Text based false news detection
--------------------------------
Faknow adopts a unique processing method for text datasets, implemented through the `TextDataset <http://127.0.0.1:8000/faknow/faknow.data.dataset.html#faknow.data.dataset.text.TextDataset>`_. This class
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