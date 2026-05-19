import torch.nn as nn
from torchvision import models


# Three backbones, all with the same interface so train.py / predict.py
# don't care which one's selected. Each replaces the ImageNet classifier
# head with a Linear layer producing `num_outputs` regression values.


class DenseNet121(nn.Module):
    def __init__(self, num_outputs=9, pretrained=True, dropout=0.0):
        super().__init__()

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
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


class DenseNet169(nn.Module):
    def __init__(self, num_outputs=9, pretrained=True, dropout=0.0):
        super().__init__()

        weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet169(weights=weights)

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


class ResNet50(nn.Module):
    def __init__(self, num_outputs=9, pretrained=True, dropout=0.0):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # ResNet uses .fc as its classifier and doesn't have a .features attribute,
        # so we patch one in for parity with the DenseNet classes.
        in_features = self.backbone.fc.in_features
        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_outputs),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.backbone(x)

    def _non_head_params(self):
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                yield param

    def freeze_backbone(self, freeze):
        for param in self._non_head_params():
            param.requires_grad = not freeze

    def head_params(self):
        return self.backbone.fc.parameters()

    def backbone_params(self):
        return self._non_head_params()


BACKBONES = {
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "resnet50": ResNet50,
}


def make_backbone(name, num_outputs=9, pretrained=True, dropout=0.0):
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {list(BACKBONES)}")
    return BACKBONES[name](num_outputs=num_outputs, pretrained=pretrained, dropout=dropout)
