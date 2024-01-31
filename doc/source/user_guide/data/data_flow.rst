Data Flow
==========
(1) **Raw data**：To make the model training uniform, FaKnow takes Dict in Python, a data structure in the form of key-value pairs, as the format of the batch input data and uses the feature name as the key and the corresponding pytorch.Tensor as the value, which allows the user to easily refer to their names to obtain the corresponding features.
(2) **Dataset**: These dataset classes included in the data module all inherit from pytorch.Dataset, and can be iteratively traversed by the __getitem__ method to obtain the above Dictdata.
(3) **Dataloder**: Users should generate the pytorch. Dataloader for data to be used. FaKnow offers a comprehensive set of Dataset classes for the built-in models, accompanied by a diverse range of data processing functionalities (e.g., text segmentation, image conversion, etc.).

