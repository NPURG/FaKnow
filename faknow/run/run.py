from faknow.run.content_based import *
from faknow.run.social_context import *
from faknow.run.knowledge_aware import *

__all__ = ['run', 'run_from_yaml']


def run(model: str, **kwargs):
    """
    run the model with the given keyword arguments

    Args:
        model(str): model name
        **kwargs: parameters to run the model
    """

    model = model.lower()
    eval(f'run_{model}(**{kwargs})')


def run_from_yaml(model: str, path: str):
    """
    run the model with the given yaml file

    Args:
        model(str): model name
        path(str): path to the yaml file
    """

    model = model.lower()
    eval(f"run_{model}_from_yaml('{path}')")
