"""Small but complete CNN / DNN / U-Net architectures.

These are intentionally small (a few 100k parameters) so the notebooks train
in a couple of minutes on CPU. They are *not* state-of-the-art -- their job
is to demonstrate end-to-end deep-learning pipelines on lensing data.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# CNN classifier
# --------------------------------------------------------------------------- #
class LensCNN(nn.Module):
    """Small VGG-style CNN classifier.

    Designed for 64x64 single-channel inputs; downsamples 64 -> 32 -> 16 -> 8
    before two fully-connected layers. Final output is a 2-class logit pair.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2, dropout: float = 0.2):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 16),
            block(16, 32),
            block(32, 64),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# Regression DNN
# --------------------------------------------------------------------------- #
class SersicRegressor(nn.Module):
    """CNN -> MLP regressor that maps a galaxy image to its Sérsic parameters.

    Output is a 7-vector (Ie, Re, n, x0, y0, e1, e2) - same convention as
    :data:`lensing.ml.datasets.PARAM_KEYS`.
    """

    def __init__(self, in_channels: int = 1, n_outputs: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# --------------------------------------------------------------------------- #
# U-Net (image-to-image)
# --------------------------------------------------------------------------- #
class _DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class UNet(nn.Module):
    """Compact U-Net (3 down/up levels) for image-to-image regression.

    Used in notebook 12 to map an observed lensed image to its source-plane
    reconstruction. Tested on 64x64 inputs; uses bilinear upsampling for
    parameter-efficiency.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16):
        super().__init__()
        self.enc1 = _DoubleConv(in_channels, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.enc3 = _DoubleConv(base * 2, base * 4)
        self.bottleneck = _DoubleConv(base * 4, base * 8)
        self.dec3 = _DoubleConv(base * 8 + base * 4, base * 4)
        self.dec2 = _DoubleConv(base * 4 + base * 2, base * 2)
        self.dec1 = _DoubleConv(base * 2 + base, base)
        self.out = nn.Conv2d(base, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out(d1)
