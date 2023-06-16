import yaml

__all__ = ['run', 'run_from_config']


def run(model: str, **kwargs):
    model = model.lower()
    eval(f'run_{model}(**kwargs)')


def run_from_config(model: str, config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    eval(f'run_{model}_from_yaml(config)')
