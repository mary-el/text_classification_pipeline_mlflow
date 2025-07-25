import yaml
import random
import glob
import os
import shutil

random.seed(0)

def random_split(folder, save_to, val_split=0.2, test_split=0.1):
    subfolders = glob.glob(folder + '/*')
    for subfolder in subfolders:
        class_name = subfolder.split('/')[-1]
        files = glob.glob(subfolder + '/*')
        random.shuffle(files)
        val_size = int(len(files) * val_split)
        test_size = int(len(files) * test_split)
        train_size = len(files) - val_size - test_size

        splits = ["train",] * train_size + ["val",] * val_size + ["test",] * test_size

        os.makedirs(f'{save_to}/train/{class_name}', exist_ok=True)
        os.makedirs(f'{save_to}/test/{class_name}', exist_ok=True)
        os.makedirs(f'{save_to}/val/{class_name}', exist_ok=True)

        for i, file in enumerate(files):
            name = os.path.basename(file)
            shutil.copyfile(file, f'{save_to}/{splits[i]}/{class_name}/{name}')


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
