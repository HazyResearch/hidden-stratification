import argparse
import collections
import os
import random
import shutil

from PIL import Image
import pandas as pd
import requests
from tqdm import tqdm

from stratification.utils.utils import flatten_dict


def main():
    parser = argparse.ArgumentParser(description='Downloads the ISIC dataset')
    parser.add_argument(
        '--root',
        default='./data/isic',
        help='Directory in which to place the `raw` and `processed` ISIC subdirectories.',
    )
    parser.add_argument(
        '--max_samples',
        default=25000,
        help='The maximum number of ISIC images to download. '
        'At time of writing there are ~23000 images in the database.',
    )
    # options for the training/validation/test split
    parser.add_argument(
        '--preset_split_path',
        default=None,
        help='If not None, generates a dataset using the ' 'split json file in the provided path',
    )
    parser.add_argument('--seed', default=1, help='The random seed used when splitting the data.')
    parser.add_argument(
        '--val_proportion',
        default=0.1,
        help='The proportion of the overall dataset to allocate '
        'to the validation partition of the dataset.',
    )
    parser.add_argument(
        '--test_proportion',
        default=0.1,
        help='The proportion of the overall dataset to allocate '
        'to the test partition of the dataset.',
    )

    args = parser.parse_args()
    root = args.root
    max_samples = args.max_samples
    preset_split_path = args.preset_split_path
    seed = args.seed
    val_proportion = args.val_proportion
    test_proportion = args.test_proportion

    print(f"Downloading data into {root} for up to {max_samples} samples...")
    print(f"Downloading metadata into {os.path.join(root, 'raw', 'metadata.csv')}...")
    download_isic_metadata(root, max_samples)
    print(f"Downloading images into {os.path.join(root, 'raw', 'images')}...")
    download_isic_images(root)
    print(
        f"Preprocessing metadata (adding columns, removing uncertain diagnoses) and saving into {os.path.join(root, 'processed', 'labels.csv')}..."
    )
    preprocess_isic_metadata(
        root,
        preset_split_path,
        seed=seed,
        val_proportion=val_proportion,
        test_proportion=test_proportion,
    )
    print(
        f"Preprocessing images (transforming to 3-channel RGB, resizing to 224x224) and saving into {os.path.join(root, 'raw', 'images')}..."
    )
    preprocess_isic_images(root)


def download_isic_metadata(root, max_samples):
    """Downloads the metadata CSV from the ISIC website."""
    raw_dir = os.path.join(root, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    r = requests.get(
        f'https://isic-archive.com/api/v1/image?limit={max_samples}'
        f'&sort=name&sortdir=1&detail=false'
    )
    image_ids = r.json()
    image_ids = [image_id['_id'] for image_id in image_ids]
    entries = []
    for image_id in tqdm(image_ids):
        r = requests.get(f'https://isic-archive.com/api/v1/image/{image_id}')
        entry = flatten_dict(r.json(), sep='.')
        entries.append(entry)

    metadata_df = pd.DataFrame(entries)
    metadata_df = metadata_df.set_index('_id')
    metadata_df.to_csv(os.path.join(raw_dir, 'metadata.csv'))
    return metadata_df


def download_isic_images(root):
    """Given the metadata CSV, downloads the ISIC images."""
    metadata_path = os.path.join(root, 'raw', 'metadata.csv')
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            'metadata.csv not downloaded. ' 'Run `download_isic_data` before this function.'
        )
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df.set_index('_id')

    raw_dir = os.path.join(root, 'raw', 'images')
    os.makedirs(raw_dir, exist_ok=True)
    image_ids = list(metadata_df.index)
    for image_id in tqdm(image_ids):
        r = requests.get(f'https://isic-archive.com/api/v1/image/{image_id}/download', stream=True)
        r.raise_for_status()
        image_path = os.path.join(raw_dir, f'{image_id}.jpg')
        with open(image_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        del r


def preprocess_isic_metadata(
    root, preset_split_path, seed=1, val_proportion=0.1, test_proportion=0.1
):
    """Preprocesses the raw ISIC metadata."""
    raw_dir = os.path.join(root, 'raw')
    processed_dir = os.path.join(root, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    metadata_path = os.path.join(raw_dir, 'metadata.csv')
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            'metadata.csv not found while preprocessing ISIC dataset. '
            'Run `download_isic_metadata` and `download_isic_images` before '
            'calling `preprocess_isic_metadata`.'
        )
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df.set_index('_id')
    labels_df = _remove_uncertain_diagnoses(metadata_df)
    labels_df = _add_split_column(
        labels_df, preset_split_path, seed, val_proportion, test_proportion
    )
    labels_df = _add_patch_column(labels_df)
    labels_df.to_csv(os.path.join(processed_dir, 'labels.csv'))


def preprocess_isic_images(root):
    """Preprocesses the images."""
    raw_dir = os.path.join(root, 'raw')
    if not os.path.isdir(os.path.join(raw_dir, 'images')):
        raise FileNotFoundError(
            'Raw ISIC images not found. Run `download_isic_images` before '
            'calling `preprocess_isic_images`.'
        )
    processed_dir = os.path.join(root, 'processed')

    labels_df = pd.read_csv(os.path.join(processed_dir, 'labels.csv'))
    labels_df = labels_df.set_index('_id')
    image_ids = labels_df.index.tolist()

    os.makedirs(os.path.join(processed_dir, 'images'), exist_ok=True)
    for image_id in tqdm(image_ids):
        out_path = os.path.join(processed_dir, 'images', f'{image_id}.jpg')
        if os.path.isfile(out_path):
            continue
        image = Image.open(os.path.join(raw_dir, 'images', f'{image_id}.jpg'))
        image = image.resize((224, 224))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(out_path)


def _remove_uncertain_diagnoses(metadata_df):
    labels_df = metadata_df.loc[
        metadata_df['meta.clinical.benign_malignant'].isin({'benign', 'malignant'})
    ]  # throw out unknowns
    print(
        f"Using {len(labels_df)} out of {len(metadata_df)} total samples with confirmed 'benign' or 'malignant' diagnoses..."
    )
    malignant_mask = labels_df['meta.clinical.benign_malignant'] == 'malignant'

    labels_df['is_malignant'] = None
    labels_df.loc[malignant_mask, 'is_malignant'] = 1
    labels_df.loc[~malignant_mask, 'is_malignant'] = 0
    assert not any(is_malignant is None for is_malignant in labels_df['is_malignant'])
    return labels_df


def _add_split_column(labels_df, preset_split_path, seed, val_proportion, test_proportion):
    """Adds a split column to the input DataFrame."""
    idxs = labels_df.index.tolist()
    if preset_split_path is not None:
        train_idxs, val_idxs, test_idxs = _get_preset_train_val_test_split(
            labels_df, preset_split_path
        )
    else:
        train_idxs, val_idxs, test_idxs = _get_random_train_val_test_split(
            idxs, seed, val_proportion, test_proportion
        )
    # add to labels_df
    labels_df['split'] = None
    labels_df.loc[train_idxs, 'split'] = 'train'
    labels_df.loc[val_idxs, 'split'] = 'val'
    labels_df.loc[test_idxs, 'split'] = 'test'
    assert not any(split is None for split in labels_df['split'])
    return labels_df


def _add_patch_column(labels_df):
    """Adds a patch column to the input DataFrame."""
    patch_mask = labels_df['dataset.name'] == 'SONIC'
    # add to labels_df
    labels_df['patch'] = None
    labels_df.loc[patch_mask, 'patch'] = 1
    labels_df.loc[~patch_mask, 'patch'] = 0
    assert not any(patch is None for patch in labels_df['patch'])
    return labels_df


def _get_preset_train_val_test_split(labels_df, preset_split_path):
    """Returns a tuple with indices for preset train, val, test, splits."""
    raise NotImplementedError


def _get_random_train_val_test_split(idxs, seed, val_proportion, test_proportion):
    """Returns a tuple with indices for random train, val, test splits."""
    n = len(idxs)
    # ensures reproducibility
    random.seed(seed)
    shuffled_idxs = random.sample(idxs, n)

    train_proportion = 1.0 - val_proportion - test_proportion
    train_n = int(train_proportion * n)
    train_idxs = shuffled_idxs[:train_n]

    val_n = int(val_proportion * n)
    val_idxs = shuffled_idxs[train_n : (train_n + val_n)]
    test_idxs = shuffled_idxs[(train_n + val_n) :]
    return (train_idxs, val_idxs, test_idxs)


if __name__ == '__main__':
    main()
