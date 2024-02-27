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

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if "vgg16" in filepath:
        model = models.vgg16(pretrained=False)
    elif "vgg13" in filepath:
        model = models.vgg13(pretrained=False)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cat_to_name = checkpoint['cat_to_name']
    return model

def process_image(image):
    (width, height) = (image.width, image.height)
    (width, height) = (256, 256*height // width)
    im_resized = image.resize((width, height))
    cropped_img = im_resized.crop((16, 16, 240, 240))
    np_image = np.array(cropped_img)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    return np_image.transpose((2,0,1))

def main(image_dir, checkpoint_dir, gpu, top_k, category_names):
    if gpu==True and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Load model
    model = load_checkpoint(checkpoint_dir)
    print(model)
    
    # Load category_names
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model.cat_to_name = cat_to_name  
    
    model.to(device)
    model.eval() 
    
    with Image.open(image_dir) as im:
        image = process_image(im)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = image.unsqueeze(0)
        image = image.to(device)
        
        with torch.no_grad():
            output = model.forward(image)

        # Calculate accuracy
        ps = torch.exp(output)
        top_p, top_class = ps.topk(top_k)
        mapping = {key: val for key, val in model.cat_to_name.items()}
        classes = [mapping[str(item+1)] for item in top_class.to('cpu').numpy().tolist()[0]]
        top_p = top_p.to('cpu').numpy().flatten()
        print(classes)
        print(top_p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command Line App - Image Classifier')
    parser.add_argument('image_dir', type=str, help='Path to an image')
    parser.add_argument('checkpoint_dir', type=str, help='Path to a checkpoint')
    parser.add_argument('--gpu', type=bool, default=True, help='GPU', required=False)
    parser.add_argument('--top_k', type=int, default=3, help='Top k most likely classes', required=False)
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names', required=False)
    
    args = parser.parse_args()
    main(args.image_dir, args.checkpoint_dir, args.gpu, args.top_k, args.category_names)