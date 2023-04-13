import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import os
import sys

parser = argparse.ArgumentParser(
    description = "To list the arguments for the flower classifier, provide the directory location of the input image as an argument. Optionally, you can specify the checkpoint save directory using --save_dir flag. For this application, you can use ImageClassifier/saved_models as the save directory. Additionally, you can specify other hyperparameters as needed. To learn more about the available hyperparameters, refer to the help documentation.",
    epilog="Thank you for using %(prog)s! We hope you enjoyed using the program :)",
)

parser.add_argument('dir',type=str)
parser.add_argument('--save_dir',type=str)
parser.add_argument('--arch',type=str)
parser.add_argument('--learning_rate',type=float)
parser.add_argument('--hidden_units',type=int)
parser.add_argument('--epochs',type=int)
parser.add_argument('--gpu',action='store_true')


args=parser.parse_args()

if os.path.isdir(args.dir):
    data_dir = args.dir
    print(f'Your Directory Was Found! {data_dir}')
else:
    print("Didn't find the specified directory. Please provide id directory path")
    sys.exit("Program is shutting down!")

if args.save_dir is None:
    print(f"Check Point will be saved in {os.getcwd()}")
    ckpt_path=os.getcwd()
else:
    if os.path.isdir(args.save_dir):
        print(f"Check Point will be saved in {args.save_dir}")
        ckpt_path=args.save_dir
    else:
        print("Provide valid path to store Checkpoint Path")
        sys.exit("Program is shutting down!")
        
    
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomGrayscale(0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}    

# Using the image datasets and the trainforms, define the dataloaders
batch_size = 32
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)
}


if args.arch is None:
    model_name='vgg16'
else:
    model_name=args.arch
model=models.vgg16(pretrained = True)
print(f"Model being used is {model_name}")

if args.hidden_units is None:
    hidden_layer=256
else:
    hidden_layer=args.hidden_units 
    

for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(
            nn.Linear(25088, hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_layer, 102),
            nn.LogSoftmax(dim = 1)
        )

model.classifier = classifier

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')
trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of trainable parameters: {trainable_total_params}')

device="cpu"
if args.gpu:
    if torch.cuda.is_available():
        print("The model is running on GPU")
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("Since there was no GPU found...")
        print("The model is running on CPU!!! This will take a significant more time, please use GPU if you are able to.")
    
criterion = nn.NLLLoss()
if args.learning_rate  is None:
    lr=0.001
else:
    lr=args.learning_rate 
    
optimizer = optim.Adam(model.classifier.parameters(), lr =lr)
model.to(device)

def train_model(model,criterion,optimizer,num_epochs=7):
    start=time.time()
    steps=0
    print_every=30
    train_loss = 0.0
    train_losses, val_losses = [], []
    train_acc,val_acc=[],[]
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if steps % print_every == 0:
                val_loss = 0
                val_accuracy = 0
                
                #setting in eval mode
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        val_loss += batch_loss.item()

                        # Calculate validation accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print( f"Training loss: {train_loss/print_every:.3f},"
                    f"Validation loss: {val_loss/len(dataloaders['valid']):.3f}, "
                    f"Validation accuracy: {val_accuracy/len(dataloaders['valid']):.3f} \n")
                train_loss = 0.0
                model.train()
    time_elapsed = time.time() - start
    print(f'Training complete!! The time for training was: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    
    return model

if args.epochs  is None:
    epochs=7
else:
    epochs=args.epochs  

model=train_model(model,criterion,optimizer,num_epochs=epochs)

checkpoint = {
    'epochs': epochs,
    'learning_rate': lr,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': criterion.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx
}

torch.save(checkpoint, os.path.join(ckpt_path,'checkpoint_model.pth'))

print('The CheckPoint Saved Successfully!')

print("Training Finished!! \n Closing program now!")