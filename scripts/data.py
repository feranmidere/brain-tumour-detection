import kagglehub
from pathlib import Path
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import shutil

def download_data(extension: str = 'Brain Tumor Data Set/Brain Tumor Data Set') -> Path:
    return Path(kagglehub.dataset_download('preetviradiya/brian-tumor-dataset', output_dir='data/')) / extension

def split_data(test_size = 0.2) -> None:
    path = download_data()
    healthy_path = path / 'Healthy'
    tumour_path = path / 'Brain Tumor'

    split_data = {}

    split_data['healthy_train'], split_data['healthy_test'] = train_test_split(list(healthy_path.iterdir()), test_size=test_size)
    split_data['tumour_train'], split_data['tumour_test'] = train_test_split(list(tumour_path.iterdir()), test_size=test_size)

    train_path = download_data('train')
    test_path = download_data('test')

    for name, paths in split_data.items():
        label, split = name.split('_')
        if split == 'train':
            new_path = (train_path / label)
        else:
            new_path = (test_path / label)
        new_path.mkdir(parents=True, exist_ok=True)

        for path in paths:
            shutil.copy(path, new_path)

if __name__ == '__main__':
    split_data()