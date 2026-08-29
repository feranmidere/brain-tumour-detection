import kagglehub
from pathlib import Path
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import shutil

def download_data(extension: str = 'Brain Tumor Data Set/Brain Tumor Data Set') -> Path:
    Path('data').mkdir(exist_ok=True)
    return Path(kagglehub.dataset_download('preetviradiya/brian-tumor-dataset', output_dir='data/')) / extension

def split_data(train_size=0.7) -> None:
    path = download_data()
    healthy_path = path / 'Healthy'
    tumour_path = path / 'Brain Tumor'

    split_data = {}

    split_data['healthy_train'], split_data['healthy_test'] = train_test_split(list(healthy_path.iterdir()), train_size=train_size, random_state=42)
    split_data['tumour_train'], split_data['tumour_test'] = train_test_split(list(tumour_path.iterdir()), train_size=train_size, random_state=42)

    split_data['healthy_test'], split_data['healthy_val'] = train_test_split(split_data['healthy_test'], train_size=0.5, random_state=42)
    split_data['tumour_test'], split_data['tumour_val'] = train_test_split(split_data['tumour_test'], train_size=0.5, random_state=42)

    for name, paths in split_data.items():
        label, split = name.split('_')
        new_path = path.parents[1] / split / label
        new_path.mkdir(parents=True, exist_ok=True)

        for current_path in paths:
            shutil.copy(current_path, new_path)

if __name__ == '__main__':
    split_data()