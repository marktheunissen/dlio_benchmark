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
import pandas as pd

from dlio_benchmark.common.constants import MODULE_DATA_READER
from utils.utility import Profile
from dlio_benchmark.reader.reader_handler import FormatReader

dlp = Profile(MODULE_DATA_READER)


class CSVReader(FormatReader):
    """
    CSV Reader reader and iterator logic.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

    @dlp.log
    def open(self, filename):
        super().open(filename)
        return pd.read_csv(filename, compression="infer").to_numpy()

    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        image = self.open_file_map[filename][sample_index]
        dlp.update(image_size=image.nbytes)

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()
    
    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True