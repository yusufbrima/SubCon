import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F  # Import the functional module for classification head

class CNNEncoder(nn.Module):
    """
    CNN-based encoder for extracting features from images using a  backbone model.

    Args:
        backbone_model (torch.nn.Module): backbone model (e.g., ResNet, DenseNet).
        latent_dim (int): Dimension of the latent space.

    Attributes:
        backbone (torch.nn.Module):  backbone model with the classification layer removed.
        fc (torch.nn.Linear): Linear layer to reduce dimensions to the specified latent_dim.
    """

    def __init__(self, backbone_model, latent_dim):
        """
        Initializes the CNN encoder.

        Args:
            backbone_model (torch.nn.Module): backbone model (e.g., ResNet, DenseNet).
            latent_dim (int): Dimension of the latent space.
        """
        super(CNNEncoder, self).__init__()

        # Use a  backbone model
        self.backbone = backbone_model(weights=None)

        # Determine the number of features in the final layer
        if 'resnet' in str(backbone_model):
            num_features = self.backbone.fc.in_features
        elif 'densenet' in str(backbone_model):
            num_features = self.backbone.classifier.in_features
        else:
            raise ValueError("Unsupported backbone model")

        # Remove the classification layer (usually the last one)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # add 1d batch norm
        self.bn = nn.BatchNorm1d(latent_dim, affine=False)
        # Add a linear layer to reduce dimensions to latent_dim
        self.fc = nn.Linear(num_features, latent_dim, nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Forward pass through the CNN encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Latent representation of the input tensor.
        """
        # Forward pass through the backbone
        features = self.backbone(x)

        # Flatten the features
        features = features.view(features.size(0), -1)

        # Forward pass through the linear layer to reduce dimensions
        latent = self.fc(features)

        return latent



class CNNClassifier(nn.Module):
    """
    CNN-based classifier for images using a backbone model.

    Args:
        backbone_model (torch.nn.Module): backbone model (e.g., ResNet, DenseNet).
        num_classes (int): Number of output classes for classification head.

    Attributes:
        backbone (torch.nn.Module): backbone model with the classification layer removed.
        classifier (nn.Linear): Classification head for predicting classes.
    """

    def __init__(self, backbone_model, num_classes):
        """
        Initializes the CNN classifier.

        Args:
            backbone_model (torch.nn.Module): backbone model (e.g., ResNet, DenseNet).
            num_classes (int): Number of output classes for classification head.
        """
        super(CNNClassifier, self).__init__()

        # Use a backbone model
        self.backbone = backbone_model(weights=None)

        # Determine the number of features in the final layer
        if 'resnet' in str(backbone_model):
            num_features = self.backbone.fc.in_features
        elif 'densenet' in str(backbone_model):
            num_features = self.backbone.classifier.in_features
        else:
            raise ValueError("Unsupported backbone model")

        # Remove the classification layer (usually the last one)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add a classification head
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass through the CNN classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Class logits.
        """
        # Forward pass through the backbone
        features = self.backbone(x)

        # Flatten the features
        features = features.view(features.size(0), -1)

        # Forward pass through the classification head
        logits = self.classifier(features)

        return logits


if __name__ == "__main__":
    pass