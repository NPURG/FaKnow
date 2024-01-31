Use run
========
If you want to quickly run a model, Faknow provides two ways to run it: Parameter Dicts and YAML Config Files.
You can create a Python file (for example, run. py) and write the following code into the file.

.. code:: python

    from faknow.run import run

    model = 'mdfend'  # lowercase short name of models
    kargs = {'train_path': 'train.json', 'test_path': 'test.json'}  # dict arguments
    run(model, **kargs)

The json file for mdfend should be like:

.. code:: json

    [
        {
            "text": "this is a sentence.",
            "domain": 9
        },
        {
            "text": "this is a sentence.",
            "domain": 1
        }
    ]

Or adopt another operating mode:

.. code:: python

    from faknow.run import run_from_yaml

    model = 'mdfend'  # lowercase short name of models
    config_path = 'mdfend.yaml'  # config file path
    run_from_yaml(model, config_path)

Your yaml config file should be like:

.. code:: yaml

    train_path: train.json # the path of training set file
    test_path: test.json # the path of testing set file

Please refer to `Config Introduction <http://127.0.0.1:8000/user_guide/config_intro.html>`_ for more details about
config settings.