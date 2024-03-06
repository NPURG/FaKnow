Image Data
==========
For image data, the MultiModalDataset class only stores the path of the image data during initialization and does not
read all the data into memory at once like text. It only reads the corresponding image file based on the given index
each time the ``getitem()`` method is called, and then converts it into a tensor using the predefined transform function.

The MultiModalDataset class adopts the design concept of reading text at once, reading images only into storage **paths**,
and only reading images into memory for processing during use. It can achieve the goal of saving memory while improving
memory management to more effectively control program memory consumption during runtime. This approach also conforms to
the common usage in Python, which is **lazy loading**, only load when needed, avoiding the problem of insufficient memory
and program crashes caused by loading all image data (especially large datasets and high-resolution images) into memory