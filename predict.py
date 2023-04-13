import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Predict.py",
        description="To list the arguments for the flower prediction, provide the directory location of the input image as an argument. Optionally, you can specify the checkpoint load directory using --load_dir flag. For this application, you can use ImageClassifier/saved_models as the load directory. Additionally, you can specify the top K classes to display using the --top_k flag, and choose to use the GPU for processing with the --gpu flag. To learn more about the available options, refer to the help documentation.",
        epilog="Thank you for using %(prog)s! We hope you enjoyed using the program :)",
    )

    parser.add_argument('img_path', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--category_names', type=str)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--gpu', action='store_true')

    return parser.parse_args()

def validate_file_path(file_path, error_message):
    if os.path.isfile(file_path):
        return file_path
    else:
        print(error_message)
        sys.exit("Program is shutting Down!!")

def get_device(args):
    if args.gpu and torch.cuda.is_available():
        print("Model is running on GPU")
        return torch.device("cuda")
    else:
        print("Model is running on CPU")
        return torch.device("cpu")

def load_checkpoint(filepath):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 102),
        nn.LogSoftmax(dim=1))

    model.classifier = classifier
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    val_test_trans = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    image = Image.open(image_path)
    return val_test_trans(image)

def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        logps = model.forward(process_image(image_path).unsqueeze(0).to(device))
        logps = logps.cpu()
        ps = torch.exp(logps)
        probs, labels = ps.topk(topk, dim=1)

        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = [class_to_idx_inv[label] for label in labels.numpy()[0]]

        return probs.numpy()[0], classes

# (Imports and functions as before...)

def main():
    args = parse_arguments()
    image_path = validate_file_path(args.img_path, "Provide valid Image Path")
    ckpt_path = validate_file_path(args.checkpoint, "Provide valid CheckPoint Path")
    device = get_device(args)
    model = load_checkpoint(ckpt_path)
    topk = args.top_k or 5
    predict_probs, predict_classes = predict(image_path, model, topk=topk, device=device)
    print(predict_probs)
    print(predict_classes)

    if args.category_names:
        cat_path = validate_file_path(args.category_names, "Provide valid category_names Path")
        with open(cat_path, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[predict_class] for predict_class in predict_classes]
        print(classes)

if __name__ == '__main__':
    main()
