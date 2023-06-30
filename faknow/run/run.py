__all__ = ['run', 'run_from_yaml']


def run(model: str, **kwargs):
    model = model.lower()
    eval(f'run_{model}(**{kwargs})')


def run_from_yaml(model: str, path: str):
    model = model.lower()
    eval(f"run_{model}_from_yaml('{path}')")
