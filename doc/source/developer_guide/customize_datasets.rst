Customize DataSets
=================
After defining a new model, if you want to define a dataset processing class that belongs to that class, there are two
ways: one is to directly select the text dataset processing class :ref:`TextDataset <faknow.data.dataset.text>` or the multimodal dataset processing class
:ref:`MultiModalDataset <faknow.data.dataset.multi_modal>` provided by Faknow, and the other is to rewrite a dataset processing class yourself, such as :ref:`safe_dataset <faknow.data.dataset.safe_dataset>`.
The *__getitem()__* method of the custom dataset class gets a sample from the dataset by index and returns the values of
each feature as a dictionary type. Specifically, for a given index index, the method returns a dictionary containing the
values of the different features in the dataset. Each feature corresponds to a key-value pair in the dictionary, where
the key is the name of the feature and the value is the value of the sample on that feature.