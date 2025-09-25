# Expose directement les classes modèles
from .face_matching_model import FaceMatchingNet
from .age_estimation_model import AgeEstimationNet
from .ocr_fraud_model import OCRFraudNet

__all__ = ["FaceMatchingNet", "AgeEstimationNet", "OCRFraudNet"]
