"""
Copyright 2024 - Chansung Park

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
import datasets
import pandas as pd
from datasets import Dataset
from huggingface_hub import create_repo
from huggingface_hub.utils import HfHubHTTPError


def push_to_hf_hub(qnas, repo_id, token, append=True):
    exist = False
    df = pd.DataFrame([qnas])
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column('target_date', datasets.features.Value('timestamp[s]'))

    try:
        create_repo(repo_id, repo_type='dataset', token=token)
    except HfHubHTTPError:
        exist = True

    if exist and append:
        existing_ds = datasets.load_dataset(repo_id)
        ds = datasets.concatenate_datasets([existing_ds['train'], ds])

    ds.push_to_hub(repo_id, token=token)
