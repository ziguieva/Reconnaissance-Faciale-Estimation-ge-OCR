import torch.nn as nn
import torchvision.models as models
import pytesseract
from PIL import Image

class OCRFraudNet(nn.Module):
    def __init__(self, num_classes=5):
        super(OCRFraudNet, self).__init__()

        # === CNN pour classification document (fraude vs normal) ===
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapter la dernière couche fully-connected à notre nombre de classes
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)

        # IMPORTANT : utiliser backbone car ton best.pth a été sauvegardé ainsi
        self.backbone = base_model

    def forward(self, x):
        return self.backbone(x)

    @staticmethod
    def extract_text(image_path, lang="eng"):
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)
        return text
