import underthesea
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VietnameseSegmenter:
    @staticmethod
    def segment(text: str) -> str:
        """
        Segments Vietnamese text and replaces spaces with underscores for multi-syllable words.
        Example: "Bệnh nhân sốt cao" -> "Bệnh_nhân sốt_cao"
        """
        if not text or not isinstance(text, str):
            return ""
        try:
            # underthesea word_tokenize with format="fixed" replaces space with underscore inside a word
            result = underthesea.word_tokenize(text, format="fixed")
            if isinstance(result, list):
                result = [w.replace(" ", "_") for w in result]
                return " ".join(result)
            return result.replace(" ", "_")
        except Exception as e:
            logger.error(f"Error in word segmentation: {e}")
            # Fallback to simple space-separated text if segmenter fails
            return text

    @staticmethod
    def segment_to_words(text: str) -> list[str]:
        """
        Segments Vietnamese text into a list of words.
        Example: "Bệnh nhân sốt cao" -> ["Bệnh_nhân", "sốt_cao"]
        """
        try:
            result = underthesea.word_tokenize(text, format="fixed")
            if isinstance(result, list):
                return [w.replace(" ", "_") for w in result]
            return [w.replace(" ", "_") for w in result.split()]
        except Exception as e:
            logger.error(f"Error in word segmentation to list: {e}")
            return [w.replace(" ", "_") for w in text.split()]
