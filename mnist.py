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
from model import CNNEncoder
from loss import BatchAllTtripletLoss
import pandas as pd
from main import train_loop

# set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.json') as json_file:
    config = json.load(json_file)
    data_folder = Path(config['dataset']['basepath'])

    fig_path =  Path(config['reporting']['figpath'])
    result_path = Path(config['reporting']['data'])
    model_path = Path(config['reporting']['model'])




if __name__ == "__main__":
    # load dataset

    # Define data transformations (you can customize these)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST training dataset
    train_dataset = datasets.MNIST(root=data_folder/'data', train=True, transform=transform, download=True)

    # Create a DataLoader for the training dataset
    batch_size = 64  # You can adjust this batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)



    # Example usage with ResNet-50 as the backbone and latent_dim=128
    latent_dim = 32
    resnet50_encoder = CNNEncoder(models.resnet50, latent_dim).to(device)


    # num_classes =  dataset._NUM_VALUES_PER_FACTOR['shape']
    # resnet50_classifier = CNNClassifier(models.resnet50, num_classes).to(device)



    criterion = BatchAllTtripletLoss() 
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet50_encoder.parameters(), lr=0.001)

    # Train the model
    num_epochs = 100  # You can adjust the number of epochs as needed
    model,train_losses = train_loop(resnet50_encoder, train_loader, criterion, optimizer, device, num_epochs)

    # model, epoch_history, loss_history, step_loss_history = train_model(resnet50_classifier, valloader, criterion, optimizer, num_epochs=100, device=device)

    # # Save the model 

    torch.save(model.state_dict(), model_path/'mnist_model.pt')


    # Create a dictionary to store the collected results
    results_dict = {
        'train_losses': train_losses
    }

    # Create a pandas DataFrame from the dictionary
    results_df = pd.DataFrame(results_dict)

    # Define the file path to write the DataFrame
    output_file = result_path/'mnist_training_results.csv'

    # Write the DataFrame to a CSV file
    results_df.to_csv(output_file, index=False)

    print(f'Results saved to {output_file}')

    print('done', train_losses)








