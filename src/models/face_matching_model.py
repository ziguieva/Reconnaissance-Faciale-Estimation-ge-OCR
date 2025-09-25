import torch
import torch.nn as nn
import torchvision.models as models

class FaceMatchingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceMatchingNet, self).__init__()

        # Backbone CNN (ResNet18 pré-entraîné)
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base_model.fc = nn.Linear(base_model.fc.in_features, embedding_dim)
        self.embedding_net = base_model

        # Similarité en sortie
        self.fc_out = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.embedding_net(x)

    def forward(self, img1, img2):
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)

        combined = torch.cat((emb1, emb2), dim=1)
        similarity = self.fc_out(combined)

        return similarity
