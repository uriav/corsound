import os
import pandas as pd
from clearml import Dataset


class MetaDataset:
    def __init__(self, raw_dataset_id, embeddings_dataset_id, metadata_csv_path):
        self.raw_dataset_path = Dataset.get(dataset_id=raw_dataset_id).get_local_copy()
        self.embeddings_dataset_path = Dataset.get(dataset_id=embeddings_dataset_id).get_local_copy()
        self.metadata = pd.read_csv(metadata_csv_path)
        self.data = self._load_data()

    def _load_data(self):
        data = {}

        # add raw dataset
        for speaker_name in os.listdir(self.raw_dataset_path):
            speaker_path = os.path.join(self.raw_dataset_path, speaker_name)
            if os.path.isdir(speaker_path):
                data[speaker_name] = {
                    'raw': [os.path.join(speaker_path, f) for f in os.listdir(speaker_path)],
                    'embeddings': [],
                    'metadata': None
                }

        # add embeddings
        for speaker_name in os.listdir(self.embeddings_dataset_path):
            speaker_path = os.path.join(self.embeddings_dataset_path, speaker_name)
            if os.path.isdir(speaker_path) and speaker_name in data:
                data[speaker_name]['embeddings'] = [os.path.join(speaker_path, f) for f in os.listdir(speaker_path)]

        # add metadata
        for _, row in self.metadata.iterrows():
            speaker_name = row['speaker_name']
            if speaker_name in data:
                data[speaker_name]['metadata'] = row.to_dict()

        return data

    def get_speaker_data(self, speaker_name):
        return self.data.get(speaker_name, None)
