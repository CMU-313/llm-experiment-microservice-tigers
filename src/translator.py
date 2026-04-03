import os

from ollama import Client

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://17313-team12.s3d.cmu.edu:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")

client = Client(host=OLLAMA_HOST)

TRANSLATION_CONTEXT = """\
You are an English translator. Translate the input text into English \
and reply only with the text translated into English. Do not include \
any explanation, notes, or extra text.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: Hello, my name is Bob.

INPUT: Können Sie mir bitte helfen?
OUTPUT: Can you please help me?
"""

CLASSIFICATION_CONTEXT = """\
You are a language classifier. Detect the language of the input text \
and reply with ONLY the English name of that language as a single word. \
Do not include any explanation, notes, or extra text.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: French

INPUT: Hello, how are you?
OUTPUT: English
"""

KNOWN_LANGUAGES = {
    "english", "french", "german", "spanish", "italian", "portuguese",
    "dutch", "russian", "chinese", "japanese", "korean", "arabic",
    "hindi", "swedish", "polish", "turkish", "greek", "czech",
    "romanian", "hungarian", "danish", "finnish", "norwegian",
    "thai", "vietnamese", "indonesian", "malay", "tagalog",
    "ukrainian", "hebrew", "persian", "bengali", "punjabi", "urdu",
    "mandarin", "cantonese", "catalan", "swahili", "tamil", "telugu",
}


def _parse_language(raw: str) -> str | None:
    """Extract a recognized language name from the LLM response.
    Returns None if the response doesn't contain a known language."""
    cleaned = raw.strip().lower()
    for lang in KNOWN_LANGUAGES:
        if lang in cleaned:
            return lang
    return None


def get_language(post: str) -> str:
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CLASSIFICATION_CONTEXT},
            {"role": "user", "content": post},
        ],
    )
    return response.message.content.strip()


def get_translation(post: str) -> str:
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": TRANSLATION_CONTEXT},
            {"role": "user", "content": post},
        ],
    )
    return response.message.content.strip()


def translate_content(content: str) -> tuple:
    """Classify the language of *content* and, if it is not English,
    translate it.

    Returns ``(is_english: bool, translated_content: str)``.

    Robustness strategy:
    - If the classifier returns gibberish (not a recognized language),
      we assume the input is English so NodeBB can still display it.
    - If the translator returns an empty or clearly broken response,
      we fall back to the original content.
    - Any exception (network error, model down, etc.) is caught and
      we assume English.
    """
    try:
        raw_lang = get_language(content)
        lang = _parse_language(raw_lang)

        if lang is None:
            return (True, content)

        is_english = lang == "english"
        translated = content
        if not is_english:
            translated = get_translation(content)
            if not translated or len(translated.strip()) == 0:
                translated = content
        return (is_english, translated)
    except Exception:
        return (True, content)