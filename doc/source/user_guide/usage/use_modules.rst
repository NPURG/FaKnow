Use Modules
===========
You can recall different modules in Faknow to satisfy your requirement.

The complete process is as follows:

.. code:: python

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

tokenizer
--------------------
.. code:: python

    max_len, bert = 170, 'bert-base-uncased'
    tokenizer = TokenizerMDFEND(max_len, bert)

dataset
--------
.. code:: python

    batch_size = 64
    train_path, test_path, validate_path = 'train.json', 'test.json', 'val.json'

    train_set = TextDataset(train_path, ['text'], tokenizer)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    validate_set = TextDataset(validate_path, ['text'], tokenizer)
    val_loader = DataLoader(validate_set, batch_size, shuffle=False)

    test_set = TextDataset(test_path, ['text'], tokenizer)
    tset_loader = DataLoader(test_set, batch_size, shuffle=False)

prepare model
--------------
.. code:: python

    domain_num = 9
    model = MDFEND(bert, domain_num)

optimizer and lr scheduler
--------------------------
.. code:: python

    lr, weight_decay, step_size, gamma = 0.00005, 5e-5, 100, 0.98
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

metrics to evaluate the model performance
------------------------------------------
.. code:: python

    evaluator = Evaluator()

train and validate
-------------------
.. code:: python

    num_epochs, device = 50, 'cpu'
    trainer = BaseTrainer(model, evaluator, optimizer, scheduler, device=device)
    trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

show test result
-----------------
.. code:: python

    print(trainer.evaluate(test_loader))