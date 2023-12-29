#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/detector.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PyTorch implementation of the final detector stage of Faster R-CNN. As input,
# takes a series of proposals (or RoIs) and produces classifications and boxes.
# The boxes are parameterized as modifications to the original incoming
# proposal boxes. That is, the proposal boxes are exactly analogous to the
# anchors that the RPN stage uses.
#

import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import RoIPool
from torchvision.models import vgg16

import numpy as np



class NoisePredictor(nn.Module):
    def __init__(self, roi_size=512, hidden_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(roi_size, hidden_dim, kernel_size=4, padding=0)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 512)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        
        return x.mean()


class DetectorNetwork(nn.Module):
  def __init__(self, num_classes, backbone, out_channels=512):
    super().__init__()

    self._input_features = 7 * 7 * backbone.feature_map_channels

    # Define network
    self._roi_pool = RoIPool(output_size = (7, 7), spatial_scale = 1.0 / backbone.feature_pixels)
    self._pool_to_feature_vector = backbone.pool_to_feature_vector
    self._classifier = nn.Linear(in_features = backbone.feature_vector_size, out_features = num_classes)
    self._regressor = nn.Linear(in_features = backbone.feature_vector_size, out_features = (num_classes - 1) * 4)
    
    # Define the noise predictor
    self._noise_predictor = NoisePredictor()
    
    # Initialize weights
    self._classifier.weight.data.normal_(mean = 0.0, std = 0.01)
    self._classifier.bias.data.zero_()
    self._regressor.weight.data.normal_(mean = 0.0, std = 0.001)
    self._regressor.bias.data.zero_()

  def forward(self, feature_map, proposals):
    """
    Predict final class and box delta regressions for region-of-interest
    proposals. The proposals serve as "anchors" for the box deltas, which
    refine the proposals into final boxes.

    Parameters
    ----------
    feature_map : torch.Tensor
      Feature map of shape (batch_size, feature_map_channels, height, width).
    proposals : torch.Tensor
      Region-of-interest box proposals that are likely to contain objects.
      Has shape (N, 4), where N is the number of proposals, with each box given
      as (y1, x1, y2, x2) in pixel coordinates.

    Returns
    -------
    torch.Tensor, torch.Tensor
      Predicted classes, (N, num_classes), encoded as a one-hot vector, and
      predicted box delta regressions, (N, 4*(num_classes-1)), where the deltas
      are expressed as (ty, tx, th, tw) and are relative to each corresponding
      proposal box. Because there is no box for the background class 0, it is
      excluded entirely and only (num_classes-1) sets of box delta targets are
      computed.
    """
    # Batch size of one for now, so no need to associate proposals with batches
    assert feature_map.shape[0] == 1, "Batch size must be 1"
    batch_idxs = t.zeros((proposals.shape[0], 1)).cuda()

    # (N, 5) tensor of (batch_idx, x1, y1, x2, y2)
    indexed_proposals = t.cat([ batch_idxs, proposals ], dim = 1)
    indexed_proposals = indexed_proposals[:, [ 0, 2, 1, 4, 3 ]] # each row, (batch_idx, y1, x1, y2, x2) -> (batch_idx, x1, y1, x2, y2)

    # RoI pooling: (N, feature_map_channels, 7, 7)
    rois = self._roi_pool(feature_map, indexed_proposals)
    
    
    #############################
    #                           #
    # Noise adversarial network #
    #                           #
    #############################
    
    # numbers of RoI
    nb_roi = rois.size(dim=0)
    
    # Initialize roi_out_noise
    rois_noise = t.zeros_like(rois).cuda()
    
    for n_roi in range(nb_roi):
      # generate noise
      noise_scale = self._noise_predictor(rois[n_roi]).cpu().detach().numpy()
      
      # noise = np.random.normal(loc=0, scale=noise_scale[0], size=rois[n_roi].cpu().shape)
    
      noise = np.random.rayleigh(scale=noise_scale, size=rois[n_roi].cpu().shape)
      noise = t.from_numpy(noise).cuda()
      
      # Update roi_out_noise
      rois_noise[n_roi] = rois[n_roi] + noise
      

    ####################################
    #                                  #
    # End of noise adversarial network #
    #                                  #
    ####################################
    

    # Forward propagate for noise loss
    y_noise = self._pool_to_feature_vector(rois = rois_noise)
    classes_raw_noise = self._classifier(y_noise)
    classes_noise = F.softmax(classes_raw_noise, dim = 1)
    box_deltas_noise = self._regressor(y_noise)
    
    # Forward propagate
    y = self._pool_to_feature_vector(rois = rois)
    classes_raw = self._classifier(y)
    classes = F.softmax(classes_raw, dim = 1)
    box_deltas = self._regressor(y)

    return classes, box_deltas, classes_noise, box_deltas_noise


def class_loss(predicted_classes, y_true):
  """
  Computes detector class loss.

  Parameters
  ----------
  predicted_classes : torch.Tensor
    RoI predicted classes as categorical vectors, (N, num_classes).
  y_true : torch.Tensor
    RoI class labels as categorical vectors, (N, num_classes).

  Returns
  -------
  torch.Tensor
    Scalar loss.
  """
  epsilon = 1e-7
  scale_factor = 1.0
  cross_entropy_per_row = -(y_true * t.log(predicted_classes + epsilon)).sum(dim = 1)
  N = cross_entropy_per_row.shape[0] + epsilon
  cross_entropy = t.sum(cross_entropy_per_row) / N
  return scale_factor * cross_entropy

def regression_loss(predicted_box_deltas, y_true):
  """
  Computes detector regression loss.

  Parameters
  ----------
  predicted_box_deltas : torch.Tensor
    RoI predicted box delta regressions, (N, 4*(num_classes-1)). The background
    class is excluded and only the non-background classes are included. Each
    set of box deltas is stored in parameterized form as (ty, tx, th, tw).
  y_true : torch.Tensor
    RoI box delta regression ground truth labels, (N, 2, 4*(num_classes-1)).
    These are stored as mask values (1 or 0) in (:,0,:) and regression
    parameters in (:,1,:). Note that it is important to mask off the predicted
    and ground truth values because they may be set to invalid values.

  Returns
  -------
  torch.Tensor
    Scalar loss.
  """
  epsilon = 1e-7
  scale_factor = 1.0
  sigma = 1.0
  sigma_squared = sigma * sigma

  # We want to unpack the regression targets and the mask of valid targets into
  # tensors each of the same shape as the predicted:
  #   (num_proposals, 4*(num_classes-1))
  # y_true has shape:
  #   (num_proposals, 2, 4*(num_classes-1))
  y_mask = y_true[:,0,:]
  y_true_targets = y_true[:,1,:]

  # Compute element-wise loss using robust L1 function for all 4 regression
  # targets
  x = y_true_targets - predicted_box_deltas
  x_abs = t.abs(x)
  is_negative_branch = (x_abs < (1.0 / sigma_squared)).float()
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # Normalize to number of proposals (e.g., 128). Although this may not be
  # what the paper does, it seems to work. Other implemetnations do this.
  # Using e.g., the number of positive proposals will cause the loss to
  # behave erratically because sometimes N will become very small.
  N = y_true.shape[0] + epsilon
  relevant_loss_terms = y_mask * losses
  return scale_factor * t.sum(relevant_loss_terms) / N

def kl_div_loss(original_output, adversarial_output):
    """
    Computes the kl divergence loss between the original distribution
    and the noisy one.

    Parameters
    ----------
    original_output : torch.Tensor
        The output of the model for the original examples, (N, num_classes).
    adversarial_output : torch.Tensor
        The output of the model for the adversarial examples, (N, num_classes).

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    original_output = F.log_softmax(original_output)
    adversarial_output = F.softmax(adversarial_output)

    Ladv = F.kl_div(original_output, adversarial_output, reduction='batchmean')
    return Ladv
