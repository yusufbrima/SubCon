import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm  # Import the tqdm module
import json 
from pathlib import Path
from sklearn.manifold import TSNE

with open('config.json') as json_file:
    config = json.load(json_file)
    data_folder = Path(config['dataset']['basepath'])

    fig_path =  Path(config['reporting']['figpath'])
    result_path = Path(config['reporting']['data'])
    model_path = Path(config['reporting']['model'])
    animation_path = Path(config['reporting']['animation'])


def create_animation(image_files, output_filename, duration=100):
    # Open and append each image to an image list
    images = []
    for image_file in image_files:
        img = Image.open(image_file)
        images.append(img)

    # Set the duration for each frame (in milliseconds)
    # You can adjust this value to control the speed of the animation
    for img in images:
        img.info['duration'] = duration

    # Save the image list as an animated GIF
    images[0].save(output_filename, save_all=True, append_images=images[1:], loop=0)
    
def extract_and_visualize_embeddings(model, dataloader, device, dim=None, counter=0, class_labels=None):
    # Set model to evaluation mode
    model.eval()

    # Initialize lists to store embeddings and labels
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)

            # Extract embeddings from the model
            if dim:
                outputs = model(inputs)
                embeddings.append(outputs.detach().cpu().numpy())
                labels.append(targets[:, dim].to(torch.long()).numpy())
            else:
                embeddings.append(model(inputs).detach().cpu().numpy())
                labels.append(targets.detach().numpy())

    # Concatenate the lists into NumPy arrays
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a scatter plot of the embeddings
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)

    # Add a legend with class labels if available
    if class_labels:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.get_cmap()(i), markersize=10,
                               label=label) for i, label in enumerate(class_labels)]
        plt.legend(handles=handles, title="Classes", loc='upper right')

    # Add counter to the title
    plt.title(f"t-SNE Embeddings (Iteration {counter})")
    plt.colorbar()

    # Save the plot as an image
    plt.savefig(animation_path / f"tsne_embeddings_{counter}.png")
    plt.close()



def plot_samples_from_loader(loader, n_rows, n_cols):
    # Get a batch of data from the loader
    data_iterator = iter(loader)
    images, labels = next(data_iterator)
    
    # Create a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5))
    
    for i in range(n_rows):
        for j in range(n_cols):
            index = i * n_cols + j
            if index < len(images):
                ax = axes[i, j]
                ax.imshow(images[index].squeeze(), cmap='gray')
                ax.set_title(f"Label: {labels[index].item()}")
                ax.axis('off')
    
    plt.subplots_adjust(wspace=0.5)
    plt.show()

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

def train_loop(model, trainloader, valloader, criterion, optimizer, device, num_epochs=10):
    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Initialize variables to track training loss for this epoch
        running_train_loss = 0.0
        num_train_batches = 0

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, labels[:,-2].to(torch.long))
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_train_loss += loss.item()
            num_train_batches += 1

            # Print progress message every, say, 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] - Batch [{batch_idx + 1}/{len(trainloader)}] - "
                      f"Train Loss: {loss.item():.4f}")

        # Calculate the average training loss for this epoch
        epoch_train_loss = running_train_loss / num_train_batches
        train_losses.append(epoch_train_loss)
        torch.save(model.state_dict(), model_path/'model.pt')
        # Validation loop
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate the loss
                loss = criterion(outputs, labels[:,-2].to(torch.long))

                # Update the running validation loss
                running_val_loss += loss.item()
                num_val_batches += 1

        # Calculate the average validation loss for this epoch
        epoch_val_loss = running_val_loss / num_val_batches
        val_losses.append(epoch_val_loss)
        
        # Print the training and validation loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    print("Training completed.")

    return model,train_losses, val_losses

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
