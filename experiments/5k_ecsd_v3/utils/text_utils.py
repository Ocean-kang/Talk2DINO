"""Text normalization helpers. Surface normalization preserves stopwords."""
import re

_WORDS = re.compile(r"[a-z0-9]+")

def canonical_surface_text(text):
    """Lowercase alphanumeric token sequence; preserves all words and their order."""
    return " ".join(_WORDS.findall(str(text).lower()))
