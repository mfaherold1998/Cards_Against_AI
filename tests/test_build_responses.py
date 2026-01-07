import unittest
import pandas as pd
from src.scripts.build_responses import split_responses, build_sentence

class TestBuildResponses(unittest.TestCase):
    def setUp(self):
        self.cards = {
            'EN': {
                'BLACK': pd.DataFrame({
                    'card_id': ['B001', 'B002'],
                    'card_text': ['This is a __ card.', 'This is another __ card with __ spaces.']
                }).set_index('card_id'),
                'WHITE': pd.DataFrame({
                    'card_id': ['W001', 'W002', 'W003'],
                    'card_text': ['test', 'funny', 'sad']
                }).set_index('card_id')
            }
        }
        self.df = pd.DataFrame({
            'response': ['This is a W001 card.', 'This is another W002 card with W003 spaces.', 'This is a card with no id.'],
            'black_id': ['B001', 'B002', 'B001'],
            'lang': ['EN', 'EN', 'EN']
        })

    def test_split_responses(self):
        """
        Test that split_responses correctly splits the DataFrame.
        """
        df_filtered, df_no_response, df_mismatch = split_responses(self.df, self.cards)
        self.assertEqual(len(df_filtered), 2)
        self.assertEqual(len(df_no_response), 1)
        self.assertEqual(len(df_mismatch), 0)

    def test_build_sentence(self):
        """
        Test that build_sentence correctly builds the sentence.
        """
        row = pd.Series({
            'lang': 'EN',
            'black_id': 'B001',
            'winners': ['W001']
        })
        sentence = build_sentence(row, self.cards)
        self.assertEqual(sentence, 'this is a test card.')

if __name__ == '__main__':
    unittest.main()
