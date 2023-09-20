from pathlib import Path
import json
import numpy as np
import h5py
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms
from dataloader import Custom3DShapesDataset,EfficientCustom3DShapesDataset
from model import CNNEncoder,CNNClassifier
from loss import BatchAllTtripletLoss
import pandas as pd

# set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.json') as json_file:
    config = json.load(json_file)
    data_folder = Path(config['dataset']['basepath'])

    fig_path =  Path(config['reporting']['figpath'])
    result_path = Path(config['reporting']['data'])
    model_path = Path(config['reporting']['model'])


def train_loop(model, trainloader, criterion, optimizer, device, num_epochs=10, dim = None):
    # Initialize a list to store training losses
    train_losses = []

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
            if dim:
                loss = criterion(outputs, labels[:, dim].to(torch.long))
            else:
                loss = criterion(outputs, labels) #.to(torch.long)
            
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
        torch.save(model.state_dict(), model_path / 'model.pt')

        # Print the training loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}")

    print("Training completed.")

    return model, train_losses


if __name__ == "__main__":
    # load dataset

    datafile = data_folder / '3dshapes.h5'

    # Initialize the custom dataset
    dataset = Custom3DShapesDataset(datafile)

    # Split the dataset into train/ validation/ test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


    # Create a PyTorch DataLoader for batching and shuffling
    batch_size = 32
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Example usage with ResNet-50 as the backbone and latent_dim=128
    latent_dim = 32
    resnet50_encoder = CNNEncoder(models.resnet50, latent_dim).to(device)


    # num_classes =  dataset._NUM_VALUES_PER_FACTOR['shape']
    # resnet50_classifier = CNNClassifier(models.resnet50, num_classes).to(device)



    criterion = BatchAllTtripletLoss() #torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet50_encoder.parameters(), lr=0.001)

    # Train the model
    num_epochs = 1  # You can adjust the number of epochs as needed
    model,train_losses = train_loop(resnet50_encoder, valloader, criterion, optimizer, device, num_epochs, dim = -2)

    # model, epoch_history, loss_history, step_loss_history = train_model(resnet50_classifier, valloader, criterion, optimizer, num_epochs=100, device=device)

    # # Save the model 

    torch.save(model.state_dict(), model_path/'model.pt')


    # Create a dictionary to store the collected results
    results_dict = {
        'train_losses': train_losses
    }

    # Create a pandas DataFrame from the dictionary
    results_df = pd.DataFrame(results_dict)

    # Define the file path to write the DataFrame
    output_file = result_path/'training_results.csv'

    # Write the DataFrame to a CSV file
    results_df.to_csv(output_file, index=False)

    print(f'Results saved to {output_file}')

    print('done', train_losses)








