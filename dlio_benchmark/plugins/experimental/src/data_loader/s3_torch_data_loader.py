"""
   Copyright (c) 2024, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from time import time
import logging
import math
import pickle
import os
import torch

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from s3torchconnector import S3MapDataset, S3ClientConfig

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import Shuffle, DatasetType, DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.data_loader.torch_data_loader import dlio_sampler
from dlio_benchmark.plugins.experimental.src.reader.s3_torch_npz_reader import S3NPZReader # Hardcoded instead of factory to pass dataset down
from dlio_benchmark.utils.utility import utcnow, DLIOMPI
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_LOADER)

# Mostly a copy of TorchDataset from dlio_benchmark/dlio_benchmark/data_loader/torch_data_loader.py

class S3TorchDataset(S3MapDataset):
    """
    Copied from TorchDataset, which currently only supports loading one sample per file.
    TODO: support multiple samples per file
    """

    @dlp.log_init
    def init(self, format_type, dataset_type, epoch, num_samples, num_workers, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch
        self.num_samples = num_samples
        self.reader = None
        self.num_images_read = 0
        self.batch_size = batch_size
        args = ConfigArguments.get_instance()
        self.serial_args = pickle.dumps(args)
        self.dlp_logger = None
        if num_workers == 0:
            self.worker_init(-1)

    @dlp.log
    def worker_init(self, worker_id):
        pickle.loads(self.serial_args)
        _args = ConfigArguments.get_instance()
        _args.configure_dlio_logging(is_child=True)
        self.dlp_logger = _args.configure_dftracer(is_child=True, use_pid=True)
        logging.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")
        self.reader = S3NPZReader(self, self.dataset_type, worker_id)

    def __del__(self):
        if self.dlp_logger:
            self.dlp_logger.finalize()

    @dlp.log
    def __len__(self):
        return self.num_samples

    @dlp.log
    def __getitem__(self, image_idx):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        logging.debug(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {image_idx} sample")
        dlp.update(step = step)
        return self.reader.read_index(image_idx, step)

    # In DLIO, the Reader is supposed to control reading the data, however S3MapDataset is the class that actually
    # has this capability for S3, so there's a bit of a weird call chain here, due to the design of DLIO vs. s3torchconnector
    def s3_get_object_reader(self, image_idx):
        return super().__getitem__(image_idx)


# Set these env vars in `env-container`` file (when using Docker), or however needed for your env.
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")

class S3TorchDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.PYTORCH)

    @dlp.log
    def read(self):
        s3config = S3ClientConfig(force_path_style=True)
        DATASET_URI = f"s3://{self._args.storage_root}/{self._args.data_folder}/{self.dataset_type.value}"
        dataset = S3TorchDataset.from_prefix(DATASET_URI,
                region=AWS_REGION,
                endpoint=S3_ENDPOINT,
                s3client_config=s3config)
        dataset.init(self.format_type, self.dataset_type, self.epoch_number, self.num_samples,
                     self._args.read_threads, self.batch_size)

        sampler = dlio_sampler(self._args.my_rank, self._args.comm_size, self.num_samples, self._args.epochs)
        if self._args.read_threads >= 1:
            prefetch_factor = math.ceil(self._args.prefetch_size / self._args.read_threads)
        else:
            prefetch_factor = self._args.prefetch_size
        if prefetch_factor > 0:
            if self._args.my_rank == 0:
                logging.debug(
                    f"{utcnow()} Prefetch size is {self._args.prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        else:
            prefetch_factor = 2
            if self._args.my_rank == 0:
                logging.debug(
                    f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")
        logging.debug(f"{utcnow()} Setup dataloader with {self._args.read_threads} workers {torch.__version__}")
        if self._args.read_threads==0:
            kwargs={}
        else:
            kwargs={'multiprocessing_context':self._args.multiprocessing_context,
                    'prefetch_factor': prefetch_factor}
            if torch.__version__ != '1.3.1':
                kwargs['persistent_workers'] = True
        if torch.__version__ == '1.3.1':
            if 'prefetch_factor' in kwargs:
                del kwargs['prefetch_factor']
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=True,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init,
                                       **kwargs)
        else:
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=True,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init,
                                       **kwargs)  # 2 is the default value
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} will read {len(self._dataset) * self.batch_size} files")

        # This line commented out in TorchDataLoader too, not sure why.
        # self._dataset.sampler.set_epoch(epoch_number)

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        step = 1
        for batch in self._dataset:
            dlp.update(step = step)
            step += 1
            yield batch
        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)

    @dlp.log
    def finalize(self):
        pass
