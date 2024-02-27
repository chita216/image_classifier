import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

def main(data_dir, save_dir, arch, lr, hidden_unit, dropout, epochs, gpu):
    if gpu==True and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_transform = transforms.Compose([transforms.RandomRotation(15),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=32, 
                                           shuffle=True,
                                           num_workers=os.cpu_count() if str(device) == "cuda" else 0,
                                           pin_memory=True if str(device) == "cuda" else False)
    valid_loader = torch.utils.data.DataLoader(valid_data, 
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=os.cpu_count() if str(device) == "cuda" else 0,
                                           pin_memory=True if str(device) == "cuda" else False)

    # Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # Train model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088, hidden_unit),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_unit, 102),
                                 nn.LogSoftmax(dim=1))
    print(model)
    
    # Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model.to(device);

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validate loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validate accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
            break
    # Save model
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'classifier': model.classifier,
              'epochs': epochs,
              'cat_to_name': cat_to_name}
    torch.save(checkpoint, save_dir + 'checkpoint_' + str(arch) + '.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command Line App - Image Classifier')
    parser.add_argument('data_dir', type=str, default='flowers', help='Path to train data')
    parser.add_argument('--save_dir', type=str, default='', help='Path to save model', required=False)
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture (vgg16, vgg13)', required=False)
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate', required=False)
    parser.add_argument('--hidden_unit', type=int, default=256, help='Hidden units', required=False)
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout', required=False)
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs', required=False)
    parser.add_argument('--gpu', type=bool, default=True, help='GPU', required=False)
    
    args = parser.parse_args()
    
    main(args.data_dir, args.save_dir, args.arch, args.lr, args.hidden_unit, args.dropout, args.epochs, args.gpu)