from pathlib import Path
import json
import numpy as np
import h5py
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from dataloader import Custom3DShapesDataset
from model import CNNEncoder,CNNClassifier
from utils import train_model
import pandas as pd

# set device 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('config.json') as json_file:
    config = json.load(json_file)
    data_folder = Path(config['dataset']['basepath'])

    fig_path =  Path(config['reporting']['figpath'])
    result_path = Path(config['reporting']['data'])
    model_path = Path(config['reporting']['model'])


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


    num_classes =  dataset._NUM_VALUES_PER_FACTOR['shape']
    resnet50_classifier = CNNClassifier(models.resnet50, num_classes).to(device)



    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet50_classifier.parameters(), lr=0.001)
    model, epoch_history, loss_history, step_loss_history = train_model(resnet50_classifier, valloader, criterion, optimizer, num_epochs=1, device=device)

    # Save the model 

    torch.save(model.state_dict(), model_path/'model.pt')


    # Create a dictionary to store the collected results
    results_dict = {
        'Epoch': epoch_history,
        'Epoch Loss': loss_history
    }

    # Create a pandas DataFrame from the dictionary
    results_df = pd.DataFrame(results_dict)

    # Define the file path to write the DataFrame
    output_file = result_path/'training_results.csv'

    # Write the DataFrame to a CSV file
    results_df.to_csv(output_file, index=False)

    print(f'Results saved to {output_file}')

    print('done', epoch_history, loss_history)








