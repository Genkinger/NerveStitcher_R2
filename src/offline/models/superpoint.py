# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from dataclasses import dataclass
from os.path import split, join
import torch
from torch import nn
from configuration import configuration
from matplotlib import pyplot as plt


@dataclass
class SuperPointConfig:
    descriptor_dim: int = 256
    nms_radius: int = 4
    max_keypoints: int = -1
    keypoint_threshold: float = 0.005
    remove_borders: int = 4


@dataclass
class SuperPointOutput:
    keypoints: list[torch.Tensor]
    scores: list[torch.Tensor]
    descriptors: list[torch.Tensor]


class SuperPointMinimal(nn.Module):

    def __init__(
        self,
        descriptor_dimensions=configuration.descriptor_dimensions,
        nms_radius=configuration.nms_radius,
        max_keypoints=configuration.max_keypoints,
        keypoint_threshold=configuration.keypoint_threshold,
        remove_borders=4,
    ):
        super().__init__()

        # CONFIG
        self.descriptor_dimensions = descriptor_dimensions
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.remove_borders = remove_borders

        # STATE
        self.scores = None
        self.keypoints = None
        self.descriptors = None
        self.encoded_representation = None
        self.batch_size = None
        self.width = None
        self.height = None
        self.descriptors_upsampled = None
    

        # MODEL DEFINITION
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.descriptor_dimensions, kernel_size=1, stride=1, padding=0
        )

        # LOAD MODEL WEIGHTS
        self.load_state_dict(
            torch.load(join(split(__file__)[0], "weights", "superpoint_v1.pth"))
        )

        self.to(configuration.device)
        print("Loaded SuperPoint model")

    def reset_state(self):
        self.scores = None
        self.keypoints = None
        self.descriptors = None
        self.encoded_representation = None
        self.width = None
        self.height = None
        self.descriptors_upsampled = None

    def encode(self, image: torch.Tensor):
        self.encoded_representation = self.relu(self.conv1a(image))
        self.encoded_representation = self.relu(
            self.conv1b(self.encoded_representation)
        )
        self.encoded_representation = self.pool(self.encoded_representation)
        self.encoded_representation = self.relu(
            self.conv2a(self.encoded_representation)
        )
        self.encoded_representation = self.relu(
            self.conv2b(self.encoded_representation)
        )
        self.encoded_representation = self.pool(self.encoded_representation)
        self.encoded_representation = self.relu(
            self.conv3a(self.encoded_representation)
        )
        self.encoded_representation = self.relu(
            self.conv3b(self.encoded_representation)
        )
        self.encoded_representation = self.pool(self.encoded_representation)
        self.encoded_representation = self.relu(
            self.conv4a(self.encoded_representation)
        )
        self.encoded_representation = self.relu(
            self.conv4b(self.encoded_representation)
        )

    def compute_scores(self):
        cPa = self.relu(self.convPa(self.encoded_representation))
        self.scores = self.convPb(cPa)
        self.scores = torch.nn.functional.softmax(self.scores, 1)[:, :-1]
        self.batch_size, _, self.height, self.width = self.scores.shape
        self.scores = self.scores.permute(0, 2, 3, 1).reshape(
            self.batch_size, self.height, self.width, 8, 8
        )
        self.scores = self.scores.permute(0, 1, 3, 2, 4).reshape(
            self.batch_size, self.height * 8, self.width * 8
        )
        self.scores = SuperPointMinimal.simple_nms(self.scores, self.nms_radius)

    def extract_keypoints(self):
        self.keypoints = []
        for s in self.scores:
            self.keypoints.append(torch.nonzero(s > self.keypoint_threshold))

    def extract_keypoint_scores(self):
        scores_out = []
        for s, k in zip(self.scores, self.keypoints):
            index = tuple(k.t())
            scores_out.append(s[index])
        self.scores = scores_out

    def remove_keypoints_at_image_border(self):
        keypoints_out, scores_out = list(
            zip(
                *[
                    SuperPointMinimal.remove_borders(
                        k, s, self.remove_borders, self.height * 8, self.width * 8
                    )
                    for k, s in zip(self.keypoints, self.scores)
                ]
            )
        )
        self.keypoints = keypoints_out
        self.scores = scores_out

    def filter_top_k_keypoints(self):
        keypoints_out, scores_out = list(
            zip(
                *[
                    SuperPointMinimal.top_k_keypoints(k, s, self.max_keypoints)
                    for k, s in zip(self.keypoints, self.scores)
                ]
            )
        )
        self.keypoints = keypoints_out
        self.scores = scores_out

    def flip_keypoints(self):
        self.keypoints = [torch.flip(k, [1]).float() for k in self.keypoints]

    def compute_descriptors(self):
        cDa = self.relu(self.convDa(self.encoded_representation))
        self.descriptors = self.convDb(cDa)
        self.descriptors = torch.nn.functional.normalize(self.descriptors, p=2, dim=1)

        sampling_grid = torch.nn.functional.affine_grid(
            torch.Tensor([[[1, 0, 0], [0, 1, 0]]]), [1, 1, self.width, self.height]
        ).to(configuration.device)
        self.descriptors_upsampled = torch.nn.functional.grid_sample(
            self.descriptors, sampling_grid, mode="bilinear", align_corners=True
        )
        self.descriptors_upsampled = torch.nn.functional.normalize(
            self.descriptors_upsampled
        )

    def extract_descriptors(self):
        cDa = self.relu(self.convDa(self.encoded_representation))
        self.descriptors = self.convDb(cDa)
        self.descriptors = torch.nn.functional.normalize(self.descriptors, p=2, dim=1)

        # Extract descriptors
        self.descriptors = [
            SuperPointMinimal.sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(self.keypoints, self.descriptors)
        ]

    def forward(self, image: torch.Tensor) -> SuperPointOutput:
        """Compute keypoints, scores, descriptors for image"""
        # Shared Encoder
        self.encode(image)
        self.compute_scores()
        self.extract_keypoints()
        self.extract_keypoint_scores()
        self.remove_keypoints_at_image_border()
        if self.max_keypoints >= 0:
            self.filter_top_k_keypoints()
        self.flip_keypoints()
        self.extract_descriptors()

        return SuperPointOutput(
            keypoints=self.keypoints, scores=self.scores, descriptors=self.descriptors
        )

    @staticmethod
    def simple_nms(scores, nms_radius: int):
        """Fast Non-maximum suppression to remove nearby points"""
        assert nms_radius >= 0

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
            )

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    @staticmethod
    def remove_borders(keypoints, scores, border: int, height: int, width: int):
        """Removes keypoints too close to the border"""
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
        mask = mask_h & mask_w
        return keypoints[mask], scores[mask]

    @staticmethod
    def top_k_keypoints(keypoints, scores, k: int):
        if k >= len(keypoints):
            return keypoints, scores
        scores, indices = torch.topk(scores, k, dim=0)
        return keypoints[indices], scores

    @staticmethod
    def sample_descriptors(keypoints, descriptors, s: int = 8):
        """Interpolate descriptors at keypoint locations"""
        b, c, h, w = descriptors.shape
        keypoints = keypoints - s / 2 + 0.5
        keypoints /= torch.tensor(
            [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
        ).to(
            keypoints
        )[None]
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        args = {"align_corners": True} if int(torch.__version__[2]) > 2 else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
        )
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1
        )
        return descriptors
