Parameters Configuration
------------------------------
Faknow supports two types of parameter configurations: Parameter Dicts and YAML Config Files.
The former receives the parameters as dict keyword arguments and the latter reads them from the yaml configuration file.

Parameter Dicts
^^^^^^^^^^^^^^^^^^
Parameter Dict is realized by the dict data structure in python, where the key
is the parameter name, and the value is the parameter value. The users can write their
parameters into a dict, and pass it into `run`.

An example is as follows:

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


Config Files
^^^^^^^^^^^^^^^^
Config Files should be organized in the format of yaml.
The users should write their parameters according to the rules aligned with
yaml, and pass them into `run_from_yaml`.

An example is as follows:

.. code:: python

    from faknow.run import run_from_yaml

    model = 'mdfend'  # lowercase short name of models
    config_path = 'mdfend.yaml'  # config file path
    run_from_yaml(model, config_path)

Your yaml config file should be like:

.. code:: yaml

    train_path: train.json # the path of training set file
    test_path: test.json # the path of testing set file