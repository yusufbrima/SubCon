import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm  # Import the tqdm module

def show_images_grid(imgs_, num_images=25, save_path=None):
    """
    Display a grid of images and optionally save the figure.

    Args:
        imgs_ (list of numpy.ndarray): List of image arrays to display.
        num_images (int, optional): Number of images to display. Defaults to 25.
        save_path (str, optional): Path to save the figure (including file extension).
            If None, the figure is not saved. Defaults to None.

    Returns:
        None
    """
    # Calculate the number of columns and rows in the grid
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))

    # Create a figure and an array of subplots for displaying the images
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            img = imgs_[ax_i]

            # Ensure the image has shape [64, 64, 3] for displaying
            if img.shape != (64, 64, 3):
                img = img.transpose((1, 2, 0))  # Transpose to [64, 64, 3]

            # Display the image
            ax.imshow(img, cmap='Greys_r', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    # Optionally save the figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Show the figure
    plt.show()

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda', dim=-2, print_freq=1000):
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        num_epochs (int): Number of training epochs (default: 10).
        device (str): Device to use for training ('cuda' or 'cpu', default: 'cuda').
        dim (int): Dimension for the labels (default: -2).
        print_freq (int): Frequency (in steps) to print and store the loss (default: 1000).

    Returns:
        model (nn.Module): Trained model.
        epoch_history (list): List of epoch numbers.
        loss_history (list): List of loss values corresponding to each epoch.
        step_loss_history (list): List of loss values corresponding to each print frequency.
    """
    model.to(device)
    model.train()  # Set the model in training mode

    epoch_history = []
    loss_history = []
    step_loss_history = []  # To store losses at each print frequency

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Wrap the train_loader with tqdm to add a progress bar
        for step, (inputs, labels) in enumerate(tqdm(train_loader), 1):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels[:, dim].to(torch.long))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % print_freq == 999:
                step_loss = running_loss / print_freq
                step_loss_history.append(step_loss)
                tqdm.write(f'Step {step}/{len(train_loader)}, Loss: {step_loss:.4f}')
                running_loss = 0.0

        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        tqdm.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')

        epoch_history.append(epoch + 1)
        loss_history.append(epoch_loss)

    print('Training complete!')
    return model, epoch_history, loss_history, step_loss_history

def create_gif_from_dataset(dataset, batch_size, fixed_factor_str, fixed_factor_value, output_filename):
    """
    Create a GIF from a dataset using specified parameters.

    Args:
        dataset (Custom3DShapesDataset): Custom dataset object.
        batch_size (int): Number of images to sample in each batch.
        fixed_factor_str (str): Name of the factor to fix (e.g., 'floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation').
        fixed_factor_value (int): Fixed value for the specified factor.
        output_filename (str): Output GIF file name.

    Returns:
        None
    """
    # Get the index of the fixed factor
    fixed_factor = dataset._FACTORS_IN_ORDER.index(fixed_factor_str)
    
    # Sample a batch of images
    img_batch = dataset.sample_batch(batch_size, fixed_factor, fixed_factor_value)

    # Convert images to PIL format
    images = [Image.fromarray((x.transpose(1, 2, 0) * 255).astype('uint8')) for x in img_batch]

    # Specify DPI (dots per inch)
    dpi = 300

    # Save the list of images as a GIF
    images[0].save(output_filename, save_all=True, append_images=images[1:], duration=100, loop=0, dpi=(dpi, dpi))
