# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from mobilesamv2 import sam_model_registry, SamPredictor


class ModelHandler:

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        mobilesamv2 = sam_model_registry['vit_h']()
        mobilesamv2.to(device=self.device)
        mobilesamv2.eval()
        encoder_path = {
            'efficientvit_l2': '/opt/nuclio/sam/l2.pt'
        }
        encoder_type = "efficientvit_l2"
        image_encoder = sam_model_registry[encoder_type](
            encoder_path[encoder_type]).to(device=self.device)
        image_encoder.eval()
        mobilesamv2.image_encoder = image_encoder
        self.latest_image = None
        self.predictor = SamPredictor(mobilesamv2)

    def handle(self, image):
        self.predictor.set_image(np.array(image))
        features = self.predictor.features
        return features
