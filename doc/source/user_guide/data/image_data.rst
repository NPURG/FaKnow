The MultiModalDataset class handles image data
----------------------------------------------
For image data, the `MultiModalDataset <http://127.0.0.1:8000/faknow/faknow.data.dataset.html#faknow.data.dataset.multi_modal.MultiModalDataset>`_ class only stores the path of the image data during initialization and does not
read all the data into memory at once like text. It only reads the corresponding image file based on the given index
each time the getitem() method is called, and then converts it into a tensor using the predefined transform function.