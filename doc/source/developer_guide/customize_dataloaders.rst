Customize DataLoaders
=================
After defining a new model, if you want to define a dataset processing class that belongs to that class, there are two
ways: one is to directly select the text dataset processing class ``TextDataset`` or the multimodal dataset processing class
``MultiModalDataset`` provided by Faknow, and the other is to rewrite a dataset processing class yourself, such as ``safe_dataset``.