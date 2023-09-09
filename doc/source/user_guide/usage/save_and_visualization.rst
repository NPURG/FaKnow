Save model and training visualization
=====================================
Currently, Faknow presents three aspects of result visualization:
`progress bar visualization <http://127.0.0.1:8000/user_guide/visualization/bar_visual.html>`_,
`log visualization <http://127.0.0.1:8000/user_guide/visualization/log_visual.html>`_,
and `TensorBoard visualization <http://127.0.0.1:8000/user_guide/visualization/tensorboard_visual.html>`_.
Among them, progress bar visualization refers to calling the tqdm library in Python to
present the training progress of the model in real-time; Log visualization refers to saving various information during
the training process in corresponding LOG log files, making it easy for users to view and understand after the training
is completed; TensorBoard visualization refers to the use of TensorBoard libraries to present the training process of a
model in the form of a graph. Both log visualization and TensorBoard visualization information files are saved in the
model running path, and the model files after training are also saved in this path. Each file and folder has a standardized
naming format, and users can open corresponding files according to their needs for information viewing and research.

For more details about training visualization, please read:

.. toctree::
   :maxdepth: 1

   ../visualization/bar_visual
   ../visualization/log_visual
   ../visualization/tensorboard_visual
