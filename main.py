import torch
import torch.nn as nn


class VGG(nn.Module):
    _cfgs = {
        'vgg_11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg_13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg_16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg_19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    def __init__(self, name, num_classes=1000, dropout=0.5):
        super().__init__()
        self.cfg = self._cfgs[name]
        self.features = self.make_layers(self.cfg)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        out = self.classifier(x)
        return out

    def make_layers(self, cfg):
        layers = []
        inp = 3
        for value in cfg:
            if value == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.Conv2d(inp, value, 3, padding=1))
                layers.append(nn.ReLU(True))
                inp = value

        return nn.Sequential(*layers)

