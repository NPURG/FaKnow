from typing import Optional, Union, Iterable, Sequence

import torch.utils.data
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t, _BaseDataLoaderIter


# you can use custom sampler here
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: Optional[bool] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None, num_workers: int = 0,
                 collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)

    def __iter__(self) -> '_BaseDataLoaderIter':
        return super().__iter__()
