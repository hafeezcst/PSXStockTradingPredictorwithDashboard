import unittest
from unittest.mock import patch, MagicMock
from src.data_processing.fair_value_calculator import FairValueCalculator

class TestFairValueCalculator(unittest.TestCase):
    @patch('requests.post')
    def test_fetch_tradingview_ta_data(self, mock_post):
        mock_post.return_value.json.return_value = {"data": "test"}
        calculator = FairValueCalculator()
        result = calculator.fetch_tradingview_ta_data("OGDC")
        self.assertIsInstance(result, dict)

if __name__ == "__main__":
    unittest.main()