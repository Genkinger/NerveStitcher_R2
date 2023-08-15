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
from pathlib import Path
from os.path import split, join
import torch
from torch import nn


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


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """

    def __init__(self, configuration: SuperPointConfig):
        super().__init__()
        self.configuration = configuration

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
            c5, self.configuration.descriptor_dim,
            kernel_size=1, stride=1, padding=0)
        # TODO(Leah): Make this less weird, separate pure data from code
        # path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(join(split(__file__)[0], "weights", "superpoint_v1.pth")))

        print('Loaded SuperPoint model')

    def encode(self, image: torch.Tensor):
        encoded_representation = self.relu(self.conv1a(image))
        encoded_representation = self.relu(self.conv1b(encoded_representation))
        encoded_representation = self.pool(encoded_representation)
        encoded_representation = self.relu(self.conv2a(encoded_representation))
        encoded_representation = self.relu(self.conv2b(encoded_representation))
        encoded_representation = self.pool(encoded_representation)
        encoded_representation = self.relu(self.conv3a(encoded_representation))
        encoded_representation = self.relu(self.conv3b(encoded_representation))
        encoded_representation = self.pool(encoded_representation)
        encoded_representation = self.relu(self.conv4a(encoded_representation))
        encoded_representation = self.relu(self.conv4b(encoded_representation))
        return encoded_representation

    def compute_scores(self, encoded_representation):
        cPa = self.relu(self.convPa(encoded_representation))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        batch_size, _, height, width = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch_size, height, width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(batch_size, height * 8, width * 8)
        scores = SuperPoint.simple_nms(scores, self.configuration.nms_radius)
        return scores, width, height

    def extract_keypoints(self, scores: torch.Tensor, threshold: float):
        keypoints = []
        for s in scores:
            keypoints.append(torch.nonzero(s > threshold))
        return keypoints

    def extract_keypoint_scores(self, scores: torch.Tensor, keypoints: list[torch.Tensor]):
        scores_out = []
        for s, k in zip(scores, keypoints):
            index = tuple(k.t())
            scores_out.append(s[index])
        return scores_out

    def remove_keypoints_at_image_border(self, keypoints: list[torch.Tensor], scores: list[torch.Tensor], width: int,
                                         height: int):
        keypoints_out, scores_out = list(zip(*[
            SuperPoint.remove_borders(k, s, self.configuration.remove_borders, height * 8, width * 8)
            for k, s in zip(keypoints, scores)]))
        return keypoints_out, scores_out

    def filter_top_k_keypoints(self, keypoints: list[torch.Tensor], scores: list[torch.Tensor], count: int):
        keypoints_out, scores_out = list(zip(*[
            SuperPoint.top_k_keypoints(k, s, count)
            for k, s in zip(keypoints, scores)]))
        return keypoints_out, scores_out

    def flip_keypoints(self, keypoints: list[torch.Tensor]):
        return [torch.flip(k, [1]).float() for k in keypoints]

    def extract_descriptors(self, encoded_representation: torch.Tensor, keypoints_flipped: list[torch.Tensor]):
        cDa = self.relu(self.convDa(encoded_representation))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [SuperPoint.sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints_flipped, descriptors)]
        return descriptors

    def forward(self, image: torch.Tensor) -> SuperPointOutput:
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        encoded_representation = self.encode(image)
        scores, width, height = self.compute_scores(encoded_representation)
        keypoints = self.extract_keypoints(scores, self.configuration.keypoint_threshold)
        scores = self.extract_keypoint_scores(scores, keypoints)
        keypoints, scores = self.remove_keypoints_at_image_border(keypoints, scores, width, height)
        if self.configuration.max_keypoints >= 0:
            keypoints, scores = self.filter_top_k_keypoints(keypoints, scores, self.configuration.max_keypoints)
        keypoints = self.flip_keypoints(keypoints)
        descriptors = self.extract_descriptors(encoded_representation, keypoints)

        return SuperPointOutput(keypoints=keypoints, scores=scores, descriptors=descriptors)

    @staticmethod
    def simple_nms(scores, nms_radius: int):
        """ Fast Non-maximum suppression to remove nearby points """
        assert (nms_radius >= 0)

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

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
        """ Removes keypoints too close to the border """
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
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        keypoints = keypoints - s / 2 + 0.5
        keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                                  ).to(keypoints)[None]
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors
