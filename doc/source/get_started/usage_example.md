# Usage Examples

## Quick Start

We provide several methods to **run integrated models** quickly with passing only few arguments. For hyperparameters
like learning rate, values from the open source code of the paper are taken as default. You can also pass your own
defined hyperparameters to these methods.

### run

You can use `run` and `run_from_yaml` methods to run integrated models. The former receives the parameters as `dict`
keyword arguments and the latter reads them from the `yaml` configuration file.

- run from kargs

```python
from faknow.run import run

model = 'mdfend'  # lowercase short name of models
kargs = {'train_path': 'train.json', 'test_path': 'test.json'}  # dict arguments
run(model, **kargs)
```

the json file for *mdfend* should be like:

```json
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
```

- run from yaml

```python
# demo.py
from faknow.run import run_from_yaml

model = 'mdfend'  # lowercase short name of models
config_path = 'mdfend.yaml'  # config file path
run_from_yaml(model, config_path)
```

your yaml config file should be like:

```yaml
# mdfend.yaml
train_path: train.json # the path of training set file
test_path: test.json # the path of testing set file
```

### run specific models

You can also run specific models using `run_model` and `run_model_from_yaml` methods by passing paramter, where `model`
is the short name of the integrated model you want to use. The usages are the same as `run` and `run_from_yaml`.
Following is an example to run *mdfend*.

```python
from faknow.run.content_based.run_mdfend import run_mdfend, run_mdfend_from_yaml

# run from kargs
kargs = {'train_path': 'train.json', 'test_path': 'test.json'}  # dict training arguments
run_mdfend(**kargs)

# or run from yaml
config_path = 'mdfend.yaml'  # config file path
run_mdfend_from_yaml(config_path)
```

## Run From Scratch

Following is an example to run *mdfend* from scratch.

```python
from faknow.data.dataset.text import TextDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.mdfend import MDFEND
from faknow.train.trainer import BaseTrainer
from faknow.run.content_based import TokenizerMDFEND

import torch
from torch.utils.data import DataLoader

# tokenizer for MDFEND
max_len, bert = 170, 'bert-base-uncased'
tokenizer = TokenizerMDFEND(max_len, bert)

# dataset
batch_size = 64
train_path, test_path, validate_path = 'train.json', 'test.json', 'val.json'

train_set = TextDataset(train_path, ['text'], tokenizer)
train_loader = DataLoader(train_set, batch_size, shuffle=True)

validate_set = TextDataset(validate_path, ['text'], tokenizer)
val_loader = DataLoader(validate_set, batch_size, shuffle=False)

test_set = TextDataset(test_path, ['text'], tokenizer)
tset_loader = DataLoader(test_set, batch_size, shuffle=False)

# prepare model
domain_num = 9
model = MDFEND(bert, domain_num)

# optimizer and lr scheduler
lr, weight_decay, step_size, gamma = 0.00005, 5e-5, 100, 0.98
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

# metrics to evaluate the model performance
evaluator = Evaluator()

# train and validate
num_epochs, device = 50, 'cpu'
trainer = BaseTrainer(model, evaluator, optimizer, scheduler, device=device)
trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

# show test result
print(trainer.evaluate(test_loader))
```
