import unittest
from meta_dataset import MetaDataset


class TestMetaDataset(unittest.TestCase):

    def setUp(self):
        self.raw_dataset_id = 'raw-dataset-uuid'
        self.embeddings_dataset_id = 'embeddings-dataset-uuid'
        self.metadata_csv_path = 'test_metadata.csv'
        self.dataset = MetaDataset(self.raw_dataset_id, self.embeddings_dataset_id, self.metadata_csv_path)

    def test_load_data(self):
        self.assertIsNotNone(self.dataset.data)

    def test_speaker_data(self):
        speaker_data = self.dataset.get_speaker_data('SPEAKER_NAME')
        self.assertIsNotNone(speaker_data)
        self.assertIn('raw', speaker_data)
        self.assertIn('embeddings', speaker_data)
        self.assertIn('metadata', speaker_data)

    def test_metadata_integration(self):
        speaker_data = self.dataset.get_speaker_data('SPEAKER_NAME')
        self.assertIsInstance(speaker_data['metadata'], dict)
        self.assertIn('gender', speaker_data['metadata'])
        self.assertIn('age', speaker_data['metadata'])


if __name__ == '__main__':
    unittest.main()
