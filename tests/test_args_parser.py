import sys
import unittest
from unittest.mock import patch
from src.utils.args_parser import get_args

class TestArgsParser(unittest.TestCase):
    @patch('sys.argv', ['1_run_llm.py', '--config-file', 'config/test_config.json'])
    def test_get_args(self):
        """
        Test that get_args returns the correct config file.
        """
        config = get_args(1)
        self.assertEqual(config['results_dir'], './tests/results')
        self.assertEqual(config['prompt_type'], 'prompt_player')
        self.assertEqual(config['cards_dir'], './data')
        self.assertEqual(config['languages'], ['EN'])
        self.assertEqual(config['file_type'], 'csv')
        self.assertEqual(config['dataset_mode'], 'test')
        self.assertEqual(config['test_num_rows'], 10)
        self.assertEqual(config['rounds'], 1)
        self.assertEqual(config['models'], ['test_model'])
        self.assertEqual(config['temperatures'], [0.7])
        self.assertEqual(config['character_description'], 'A test character.')

if __name__ == '__main__':
    unittest.main()
