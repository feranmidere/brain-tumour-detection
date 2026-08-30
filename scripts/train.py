import torch.nn as nn
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from scripts.data import download_data
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.optim as optim
from pathlib import Path
import os
import pandas as pd
from huggingface_hub import hf_hub_download

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_transforms() -> tuple[v2.Compose, v2.Compose]:
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.RGB(),
        v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        v2.Resize(224),
        v2.CenterCrop((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(sigma = 0.01),
        v2.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = v2.Compose([
        v2.ToImage(),
        v2.RGB(),
        v2.Resize(224),
        v2.CenterCrop((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transforms, valid_transforms

def make_dataset(set: str, transforms) -> ImageFolder:
    data = ImageFolder(
        root = download_data(set),
        transform = transforms
    )
    return data    


def train(model, train_data, valid_data, epochs=15, optimiser=optim.Adam, early_stopping=3, optimiser_args: dict=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.fc = nn.Linear(in_features=model.fc.in_features,  out_features=1)  
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimiser_args is None:
        opt = optimiser(trainable_params)   
    else:
        opt = optimiser(trainable_params, **optimiser_args)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32)

    loss_function = nn.BCEWithLogitsLoss()

    losses = {'train': [], 'val': []}
    metrics = {'train': [], 'val': []}

    best_val_loss = float('inf')
    no_decrease_count = 0
    for epoch in tqdm(range(epochs)):
        running_train_loss = 0
        running_train_correct = 0
        model.train()
        for train_batch in tqdm(train_loader):
            train_batch_X, train_batch_y = train_batch[0].to(device), train_batch[1].unsqueeze(1).float().to(device)

            model.zero_grad()
            opt.zero_grad()

            train_output = model.forward(train_batch_X)
            train_loss = loss_function(train_output, train_batch_y)

            train_output_class = (train_output >= 0).int()
            train_correct = (train_output_class == train_batch_y).int().sum()

            running_train_correct += train_correct.item()
            running_train_loss += train_loss.item() * len(train_batch_y)

            train_loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            running_valid_loss = 0
            running_valid_correct = 0
            for valid_batch in tqdm(valid_loader):
                valid_batch_X, valid_batch_y = valid_batch[0].to(device), valid_batch[1].unsqueeze(1).float().to(device)

                valid_output = model.forward(valid_batch_X)
                valid_loss = loss_function(valid_output, valid_batch_y)

                valid_output_class = (valid_output >= 0).int()
                valid_correct = (valid_output_class == valid_batch_y).int().sum()

                running_valid_correct += valid_correct.item()
                running_valid_loss += valid_loss.item() * len(valid_batch_y)

        avg_train_loss = running_train_loss / len(train_data)
        avg_val_loss = running_valid_loss / len(valid_data)
        avg_train_accuracy = running_train_correct / len(train_data)
        avg_val_accuracy = running_valid_correct / len(valid_data)

        losses['train'].append(avg_train_loss)
        losses['val'].append(avg_val_loss)
        metrics['train'].append(avg_train_accuracy)
        metrics['val'].append(avg_val_accuracy)

        print(f'Epoch {epoch + 1}: Training accuracy: {avg_train_accuracy}  Validation accuracy: {avg_val_accuracy}')

        if avg_val_loss >= best_val_loss:
            no_decrease_count += 1
        else:
            best_val_loss = avg_val_loss
            no_decrease_count = 0

        if no_decrease_count >= early_stopping:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

    return losses, metrics

def run_training(train_args: dict = None, return_results: bool=True):
    in_dir = Path('base_model')
    in_dir.mkdir(exist_ok=True)
    model_path = hf_hub_download(repo_id='Lab-Rasool/RadImageNet', filename='ResNet50.pt', local_dir=in_dir)
    model = torch.load(model_path)
    train_transforms, valid_transforms = set_transforms()
    train_data, valid_data = make_dataset('train', train_transforms), make_dataset('val', valid_transforms)
    if train_args is None:
        losses, metrics = train(train_data, valid_data)
    else:
        losses, metrics = train(train_data, valid_data, **train_args)
    if return_results:
        return model, losses, metrics
    else:
        cwd = Path(os.getcwd())
        out_dir = cwd.parent / 'model'
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / 'model.pt')
        pd.DataFrame(losses).to_csv(out_dir / 'losses.csv')
        pd.DataFrame(metrics).to_csv(out_dir / 'metrics.csv')

if __name__ == '__main__':
    model, losses, metrics = run_training()