"""
Unit tests for tokenizer.py
"""

from app.tokenizer import TokenCounter, LlamaVocabTokenCounter


class TestTokenCounterProtocol:
    def test_protocol_exists(self):
        """TokenCounter protocol should be defined."""
        assert hasattr(TokenCounter, 'count')
        assert callable(TokenCounter.count)


class TestLlamaVocabTokenCounter:
    def test_mock_implementation(self, mock_tokenizer):
        """Test the mock tokenizer fixture works."""
        assert mock_tokenizer.count("hello") == 1
        assert mock_tokenizer.count("hello world") == 2
        assert mock_tokenizer.count("") == 0
        assert mock_tokenizer.count("a" * 100) == 25

    def test_count_special_chars(self, mock_tokenizer):
        """Token counting handles special characters."""
        assert mock_tokenizer.count("hello\nworld") >= 2
        assert mock_tokenizer.count("🎉") >= 1
        assert mock_tokenizer.count("hello\tworld") >= 2