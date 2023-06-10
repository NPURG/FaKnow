from typing import Dict, Any


def run(config: Dict[str, Any]):
    pass
    # 包括对dataset的参数：tokenizer transform
    # dataloader的参数： batch size
    # dataset = create_dataset(config['data'])
    # model_name = config['model']['name']
    # model = get_model(config['model'])
    # trainer = get_trainer(config['train'], model)
    #
    # if model_name == 'eann':
    #     runner.run_eann(config)
