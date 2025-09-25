import torch
import torch.nn as nn
import torchvision.models as models

class AgeEstimationNet(nn.Module):
    def __init__(self):
        super(AgeEstimationNet, self).__init__()

        # ResNet18 pré-entraîné
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modifier la dernière couche → sortie = 1 (âge estimé)
        base_model.fc = nn.Linear(base_model.fc.in_features, 1)

        self.model = base_model

    def forward(self, x):
        return self.model(x).squeeze(1)  # sortie: âge estimé
