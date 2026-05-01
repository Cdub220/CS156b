import torch.nn as nn
from torchvision import models




class DenseNet121(nn.Module):
    def __init__(self, num_outputs=9, pretrained=True, dropout=0.0):
        super().__init__()

        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.densenet121(weights=weights)

        in_features = self.backbone.classifier.in_features

        if dropout > 0:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_outputs),
            )
        else:
            self.backbone.classifier = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self, freeze):
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze

    def head_params(self):
        return self.backbone.classifier.parameters()

    def backbone_params(self):
        return self.backbone.features.parameters()
