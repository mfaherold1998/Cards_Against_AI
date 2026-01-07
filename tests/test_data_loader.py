import unittest
import pandas as pd
from src.data.data_loader import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        """
        Test that load_data returns the correct DataFrame.
        """
        df, errors = load_data('data/cards_test_data.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['card_id', 'card_text'])
        self.assertEqual(errors, [])

if __name__ == '__main__':
    unittest.main()
