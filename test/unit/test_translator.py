from src.translator import translate_content
from unittest.mock import patch, MagicMock

from src.translator import translate_content, client

def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message."

def test_llm_normal_response():
    is_english, translated_content = translate_content("The weather today is absolutely beautiful.")
    assert is_english == True
    assert translated_content == "The weather today is absolutely beautiful."

def test_llm_gibberish_response():
    is_english, translated_content = translate_content("asdkjhqwer zxcvbnm poiuyt!!!")
    assert is_english == True
    assert translated_content == "asdkjhqwer zxcvbnm poiuyt!!!"

def _mock_response(content: str):
    """Helper: build a fake Ollama chat response."""
    resp = MagicMock()
    resp.message.content = content
    return resp


# ---------- normal-path tests ----------

def test_llm_normal_response():
    """Non-English text is detected and translated correctly."""
    with patch.object(client, "chat") as mock_chat:
        mock_chat.side_effect = [
            _mock_response("German"),
            _mock_response("Here is your first example."),
        ]

        is_english, translated = translate_content("Hier ist dein erstes Beispiel.")

        assert is_english is False
        assert translated == "Here is your first example."
        assert mock_chat.call_count == 2


def test_llm_english_detection():
    """English text is identified and returned unchanged."""
    with patch.object(client, "chat") as mock_chat:
        mock_chat.return_value = _mock_response("English")

        is_english, translated = translate_content("Hello, how are you?")

        assert is_english is True
        assert translated == "Hello, how are you?"
        assert mock_chat.call_count == 1


# ---------- error / gibberish tests ----------

def test_llm_gibberish_response():
    """If the classifier returns unrecognizable gibberish, assume English
    so the post still displays normally."""
    with patch.object(client, "chat") as mock_chat:
        mock_chat.return_value = _mock_response("I don't understand your request")

        is_english, translated = translate_content("Hier ist dein erstes Beispiel.")

        assert is_english is True
        assert translated == "Hier ist dein erstes Beispiel."
        assert mock_chat.call_count == 1


def test_llm_exception():
    """If the LLM throws an exception the function still returns gracefully."""
    with patch.object(client, "chat") as mock_chat:
        mock_chat.side_effect = Exception("Model unavailable")

        is_english, translated = translate_content("Hier ist dein erstes Beispiel.")

        assert is_english is True
        assert translated == "Hier ist dein erstes Beispiel."


def test_llm_timeout():
    """A connection/timeout error is handled without crashing."""
    with patch.object(client, "chat") as mock_chat:
        mock_chat.side_effect = ConnectionError("Connection refused")

        is_english, translated = translate_content("Bonjour le monde")

        assert is_english is True
        assert translated == "Bonjour le monde"


def test_llm_empty_translation():
    """If the translator returns an empty string, fall back to original content."""
    with patch.object(client, "chat") as mock_chat:
        mock_chat.side_effect = [
            _mock_response("French"),
            _mock_response(""),
        ]

        is_english, translated = translate_content("Bonjour le monde")

        assert is_english is False
        assert translated == "Bonjour le monde"


def test_llm_empty_input():
    """Empty string input does not crash the translator."""
    with patch.object(client, "chat") as mock_chat:
        mock_chat.return_value = _mock_response("English")

        is_english, translated = translate_content("")

        assert isinstance(is_english, bool)
        assert isinstance(translated, str)