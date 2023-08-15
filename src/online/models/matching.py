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

import torch
from dataclasses import dataclass
from .superpoint import SuperPoint, SuperPointInput, SuperPointOutput, SuperPointConfig
from .superglue import SuperGlue, SuperGlueInput, SuperGlueOutput, SuperGlueConfig


@dataclass
class MatchingInput:
    image0: torch.Tensor
    image1: torch.Tensor
    index: int


@dataclass
class MatchingOutput:
    keypoints0: torch.Tensor
    keypoints1: torch.Tensor
    matches0: torch.Tensor
    matches1: torch.Tensor
    matching_scores0: torch.Tensor
    matching_scores1: torch.Tensor


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """

    def __init__(self, superpoint_config: SuperPointConfig, superglue_config: SuperGlueConfig):
        super().__init__()
        self.superpoint = SuperPoint(superpoint_config)
        self.superglue = SuperGlue(superglue_config)

    def forward(self, data: MatchingInput):
        super_point_result_0: SuperPointOutput = self.superpoint(SuperPointInput(image=data.image0))
        super_point_result_1: SuperPointOutput = self.superpoint(SuperPointInput(image=data.image1))

        superglue_input = SuperGlueInput(image0=data.image0,
                                         image1=data.image1,
                                         keypoints0=torch.stack(super_point_result_0.keypoints),
                                         keypoints1=torch.stack(super_point_result_1.keypoints),
                                         descriptors0=torch.stack(super_point_result_0.descriptors),
                                         descriptors1=torch.stack(super_point_result_1.descriptors),
                                         scores0=torch.stack(super_point_result_0.scores),
                                         scores1=torch.stack(super_point_result_1.scores),
                                         index=data.index)

        superglue_result: SuperGlueOutput = self.superglue(superglue_input, data.index)

        return MatchingOutput(**vars(superglue_result), keypoints0=superglue_input.keypoints0,
                              keypoints1=superglue_input.keypoints1)
