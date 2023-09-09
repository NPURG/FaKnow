Data Flow
==========
Faknow is mainly composed of three modules: `data module <http://127.0.0.1:8000/user_guide/data_intro.html>`_, `algorithm
model module <http://127.0.0.1:8000/user_guide/model_intro.html>`_, and
`training and validation module <http://127.0.0.1:8000/user_guide/train_eval_intro.html>`_.
The data module is responsible for preprocessing the user input data and feeding it into the algorithm model module;
There are several classic false information recognition algorithms in the algorithm model module; The data is trained
through an embedded algorithm model, and the performance of the algorithm model is evaluated by verifying its various
indicators. Finally, the final algorithm model is saved.

.. image:: ../../media/data_flow.png
    :align: center


(1) Users input raw data into the data module. In this module, the original data is transformed into PyTorch tensor format through the pre-processing class of the data set designed in advance, and the data in tensor format is finally packaged through the DataLoader class.
    For more details about data pre-processing, please read `Text based false news detection <http://127.0.0.1:8000/user_guide/data/text_based.html>`_ and `Multimodal information based false news detection <http://127.0.0.1:8000/user_guide/data/multi_info_based.html>`_.

(2) The user selects the model in the algorithm model module and sends the data packaged in the data module to the selected algorithm model.
    For more details about model, please read `Model Introduction <http://127.0.0.1:8000/user_guide/model_intro.html>`_.

(3) Each built-in algorithm model calls the common trainers and validators in the training and validation module. After the user selects the model, loss_func, optimizer, evaluator, scheduler and determines the hyperparameters, the model begins to train and validate the data.
    For more details about training and validation, please read `Training & Evaluation Introduction <http://127.0.0.1:8000/user_guide/train_eval_intro.html>`_.

(4) The process in the training and validation module is presented in the form of progress bars, LOG logs, and TensorBoard, and the model is ultimately saved.
    For more details about training and validation, please read `Save model and training visualization <http://127.0.0.1:8000/user_guide/usage/save_and_visualization.html>`_.
