import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy as sp
from pytorch_metric_learning import losses
from math import log
eps = 1e-8 # a small number to prevent division by zero

def get_triplet_mask(labels):
  """compute a mask for valid triplets
  Args:
    labels: Batch of integer labels. shape: (batch_size,)
  Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
  """
  # step 1 - get a mask for distinct indices

  # shape: (batch_size, batch_size)
  indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
  indices_not_equal = torch.logical_not(indices_equal)
  # shape: (batch_size, batch_size, 1)
  i_not_equal_j = indices_not_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_not_equal_k = indices_not_equal.unsqueeze(1)
  # shape: (1, batch_size, batch_size)
  j_not_equal_k = indices_not_equal.unsqueeze(0)
  # Shape: (batch_size, batch_size, batch_size)
  distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

  # step 2 - get a mask for valid anchor-positive-negative triplets

  # shape: (batch_size, batch_size)
  labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
  # shape: (batch_size, batch_size, 1)
  i_equal_j = labels_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_equal_k = labels_equal.unsqueeze(1)
  # shape: (batch_size, batch_size, batch_size)
  valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

  # step 3 - combine two masks
  mask = torch.logical_and(distinct_indices, valid_indices)

  return mask


class BatchAllTtripletLoss(nn.Module):
  """Uses all valid triplets to compute Triplet loss
  Args:
    margin: Margin value in the Triplet Loss equation
  """
  def __init__(self, margin=1.):
    super().__init__()
    self.margin = margin
    
  def forward(self, embeddings, labels):
    """computes loss value.
    Args:
      embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
      labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
    Returns:
      Scalar loss value.
    """

    print("before distance matrix")
    # step 1 - get distance matrix
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    print("after distance matrix")

    # print('distance_matrix', distance_matrix)

    # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

    # shape: (batch_size, batch_size, 1)
    anchor_positive_dists = distance_matrix.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    anchor_negative_dists = distance_matrix.unsqueeze(1)
    # get loss values for all possible n^3 triplets
    # shape: (batch_size, batch_size, batch_size)
    triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

    # step 3 - filter out invalid or easy triplets by setting their loss values to 0

    # shape: (batch_size, batch_size, batch_size)
    mask = get_triplet_mask(labels)
    triplet_loss *= mask
    # easy triplets have negative loss values
    triplet_loss = F.relu(triplet_loss)

    # step 4 - compute scalar loss value by averaging positive losses
    num_positive_losses = (triplet_loss > eps).float().sum()
    triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

    return triplet_loss

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=1.0):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, embeddings, labels):
        device = embeddings.device

        # Compute cosine similarity between embeddings 
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        
        # Create diagonal mask to ignore self-similarities
        batch_size = embeddings.shape[0]
        mask = torch.eye(batch_size, device=device).bool()
        
        # Positive pairs are those with same labels, negative otherwise
        positive_pairs = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~mask
        negative_pairs = ~positive_pairs & ~mask

        # Compute logarithm of positive cosine similarities
        log_pos = torch.log(torch.clamp(cosine_sim[positive_pairs], min=1e-6))
        
        # Compute logarithm of negative cosine similarities
        log_neg = torch.log(torch.clamp(self.margin - cosine_sim[negative_pairs], min=1e-6))
        
        # Compute mean of logarithmic similarities
        positive_loss = -log_pos.mean()
        negative_loss = -log_neg.mean()
        
        # Combine losses
        loss = positive_loss + negative_loss
        
        return loss
  
# class SupervisedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         """
#         Initialize the Supervised Contrastive Loss.

#         Args:
#             temperature (float): A hyperparameter that controls the scaling of the logits.
#                 A higher temperature makes similarity scores more uniform, while a lower
#                 temperature sharpens the differences.
#         """
#         super(SupervisedContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, feature_vectors, labels):
#         """
#         Compute the Supervised Contrastive Loss.

#         Args:
#             feature_vectors (Tensor): The feature vectors to be embedded and compared.
#                 Shape: (batch_size, embedding_dimension).
#             labels (Tensor): The corresponding class labels for each sample.
#                 Shape: (batch_size,).

#         Returns:
#             Tensor: The computed loss value.
#         """
#         # Normalize feature vectors
#         feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)

#         # Compute similarity scores (logits)
#         logits = torch.div(
#             torch.matmul(
#                 feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
#             ),
#             self.temperature,
#         )

#         # Calculate the supervised contrastive loss using NTXentLoss
#         # Note: The implementation of NTXentLoss is not provided here.
#         # It is assumed to be an external loss function.
#         # Adjust the temperature value as needed.
#         return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))


# class SupervisedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         """
#         Implementation of the loss described in the paper Supervised Contrastive Learning :
#         https://arxiv.org/abs/2004.11362

#         :param temperature: int

#         Credits: https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
#         """
#         super(SupervisedContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, projections, targets):
#         """

#         :param projections: torch.Tensor, shape [batch_size, projection_dim]
#         :param targets: torch.Tensor, shape [batch_size]
#         :return: torch.Tensor, scalar
#         """
#         device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

#         dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
#         # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
#         exp_dot_tempered = (
#             torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
#         )

#         mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
#         mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
#         mask_combined = mask_similar_class * mask_anchor_out
#         cardinality_per_samples = torch.sum(mask_combined, dim=1)

#         log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
#         supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
#         supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

#         return supervised_contrastive_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR 
    Code from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
if __name__ == "__main__":
    pass