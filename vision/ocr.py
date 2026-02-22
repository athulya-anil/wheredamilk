"""
vision/ocr.py — EasyOCR wrapper with YOLO integration.

EasyOCR provides high-quality text recognition comparable to Google ML Kit.
Supports 80+ languages and handles rotated text well.

See: https://github.com/JaidedAI/EasyOCR
"""

import numpy as np
import easyocr


class OCRReader:
    def __init__(self, languages=['en'], gpu=False):
        """Initialize EasyOCR reader.
        
        Args:
            languages: List of language codes (default: English only)
            gpu: Set to True if you have CUDA; False for CPU
        """
        self._reader = easyocr.Reader(languages, gpu=gpu)
        print(f"[ocr] EasyOCR ready")

    def read_text(self, frame, box: dict) -> str:
        """
        Crop `frame` to the region defined by `box` and run OCR.
        Returns joined string of all detected text, or ''.
        """
        x1 = max(0, box["x1"])
        y1 = max(0, box["y1"])
        x2 = min(frame.shape[1], box["x2"])
        y2 = min(frame.shape[0], box["y2"])

        if x2 <= x1 or y2 <= y1:
            return ""

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        results = self._reader.readtext(crop, detail=1)
        texts = []
        
        if results:
            for result in results:
                try:
                    # result = (bbox, text, confidence)
                    bbox, text, conf = result
                    if conf > 0.3:  # Filter low confidence
                        texts.append(text)
                except (TypeError, ValueError):
                    pass
        
        return " ".join(texts).strip()

    def read_text_with_confidence(self, frame, box: dict) -> tuple[str, float]:
        """
        Crop `frame` to the region defined by `box` and run OCR.
        Returns tuple of (text_string, average_confidence) or ('', 0.0).
        """
        x1 = max(0, box["x1"])
        y1 = max(0, box["y1"])
        x2 = min(frame.shape[1], box["x2"])
        y2 = min(frame.shape[0], box["y2"])

        if x2 <= x1 or y2 <= y1:
            return "", 0.0

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "", 0.0

        results = self._reader.readtext(crop, detail=1)
        texts = []
        confidences = []
        
        if results:
            for result in results:
                try:
                    bbox, text, conf = result
                    if conf > 0.3:
                        texts.append(text)
                        confidences.append(conf)
                except (TypeError, ValueError):
                    pass
        
        avg_conf = np.mean(confidences) if confidences else 0.0
        return " ".join(texts).strip(), float(avg_conf)

    def enrich_detections(self, frame, boxes: list[dict]) -> list[dict]:
        """
        Process multiple YOLO detection boxes with OCR.
        Returns enriched boxes with 'text' and 'text_conf' fields.
        """
        enriched = []
        for box in boxes:
            text, conf = self.read_text_with_confidence(frame, box)
            box_copy = box.copy()
            box_copy["text"] = text
            box_copy["text_conf"] = conf
            enriched.append(box_copy)
        return enriched

