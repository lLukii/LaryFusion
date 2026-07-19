# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class ResidualBlock(nn.Module):
  def __init__(self, label_dim, residual_channels, dilation, uncond=False):
    '''
    :param label_dim: dimension of the shared label embedding (dlabel in the paper)
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable class-conditional embedding
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    if not uncond: # conditional model
      # layer-specific Conv1x1 mapping the shared label embedding to 2C channels
      self.conditioner_projection = Conv1d(label_dim, 2 * residual_channels, 1)
    else: # unconditional model
      self.conditioner_projection = None

    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, conditioner=None):
    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    y = self.dilated_conv(y)
    if self.conditioner_projection is not None: # using a class-conditional model
      # broadcast the per-class embedding as a bias term after the dilated conv
      conditioner = self.conditioner_projection(conditioner.unsqueeze(-1))
      y = y + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
  def __init__(self, params, n_classes=2, label_dim=128):
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(1, params.residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
    if self.params.unconditional: # use unconditional model
      self.label_embedding = None
    else:
      # shared embedding table (dlabel = 128 per the paper)
      self.label_embedding = nn.Embedding(n_classes, label_dim)

    self.residual_layers = nn.ModuleList([
        ResidualBlock(label_dim, params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
        for i in range(params.residual_layers)
    ])
    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)

  def forward(self, audio, diffusion_step, label=None):
    assert (label is None and self.label_embedding is None) or \
           (label is not None and self.label_embedding is not None)
    x = audio.unsqueeze(1)
    x = self.input_projection(x)
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)
    if self.label_embedding is not None: # use conditional model
      label = self.label_embedding(label)

    skip = None
    for layer in self.residual_layers:
      x, skip_connection = layer(x, diffusion_step, label)
      skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x
