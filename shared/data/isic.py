from __future__ import annotations

import logging
import os
import shutil
import zipfile
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Callable, ClassVar, NamedTuple, TypeVar, Union

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.tensor import Tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm

from shared.configs.enums import IsicAttrs
from shared.utils.utils import flatten_dict

__all__ = ["IsicDataset"]

LOGGER = logging.getLogger(__name__)


class Sample(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor


Transform = Callable[[Union[Image.Image, Tensor]], Tensor]
T = TypeVar("T")


class IsicDataset(Dataset):
    """PyTorch Dataset for the ISIC 2018 dataset from
    'Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International
    Skin Imaging Collaboration (ISIC)',"""

    _pbar_col: ClassVar[str] = "#fac000"
    _rest_api_url: ClassVar[str] = "https://isic-archive.com/api/v1"

    def __init__(
        self,
        root: str | Path,
        download: bool = True,
        max_samples: int = 25_000,  # default is the number of samples used for the NSLB paper
        sens_attr: IsicAttrs = IsicAttrs.histo,
        target_attr: IsicAttrs = IsicAttrs.malignant,
        transform: Transform | None = ToTensor(),
        target_transform: Transform | None = None,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.download = download
        self._data_dir = self.root / "ISIC"
        self._processed_dir = self._data_dir / "processed"
        self._raw_dir = self._data_dir / "raw"

        if max_samples < 1:
            raise ValueError("max_samples must be a positive integer.")
        self.max_samples = max_samples
        if self.download:
            self._download_data()
            self._preprocess_data()
        elif not self._check_downloaded():
            raise RuntimeError(
                f"Data don't exist at location {self._processed_dir.resolve()}. "
                "Have you downloaded it?"
            )

        self.metadata = pd.read_csv(self._processed_dir / "labels.csv")
        # Divide up the dataframe into it's constituent arrays because indexing with pandas is
        # considerably slower than indexing with numpy/torch
        self.x = self.metadata["path"].values
        self.s = torch.as_tensor(self.metadata[sens_attr.name], dtype=torch.int32)
        self.y = torch.as_tensor(self.metadata[target_attr.name], dtype=torch.int32)

        self.transform = transform
        self.target_transform = target_transform

    def _check_downloaded(self) -> bool:
        return (self._raw_dir / "images").exists() and (self._raw_dir / "metadata.csv").exists()

    def _check_processed(self) -> bool:
        return (self._processed_dir / "ISIC-images").exists() and (
            self._processed_dir / "labels.csv"
        ).exists()

    @staticmethod
    def chunk(it: Iterable[T], size: int) -> Iterator[list[T]]:
        """Divide any iterable into chunks of the given size."""
        it = iter(it)
        return iter(lambda: list(islice(it, size)), [])  # this is magic from stackoverflow

    def _download_isic_metadata(self) -> pd.DataFrame:
        """Downloads the metadata CSV from the ISIC website."""
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        req = requests.get(
            f"{self._rest_api_url}/image?limit={self.max_samples}"
            f"&sort=name&sortdir=1&detail=false"
        )
        image_ids = req.json()
        image_ids = [image_id["_id"] for image_id in image_ids]

        template_start = "?limit=300&sort=name&sortdir=1&detail=true&imageIds=%5B%22"
        template_sep = "%22%2C%22"
        template_end = "%22%5D"
        entries = []
        with tqdm(
            total=(len(image_ids) - 1) // 300 + 1,
            desc="Downloading metadata",
            colour=self._pbar_col,
        ) as pbar:
            for block in self.chunk(image_ids, 300):
                pbar.set_postfix(image_id=block[0])
                args = ""
                args += template_start
                args += template_sep.join(block)
                args += template_end
                req = requests.get(f"{self._rest_api_url}/image{args}")
                image_details = req.json()
                for image_detail in image_details:
                    entry = flatten_dict(image_detail, sep=".")
                    entries.append(entry)
                pbar.update()

        metadata_df = pd.DataFrame(entries)
        metadata_df = metadata_df.set_index("_id")
        metadata_df.to_csv(self._raw_dir / "metadata.csv")
        return metadata_df

    def _download_isic_images(self) -> None:
        """Given the metadata CSV, downloads the ISIC images."""
        metadata_path = self._raw_dir / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "metadata.csv not downloaded. " "Run 'download_isic_data` before this function."
            )
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df.set_index("_id")

        template_start = "?include=images&imageIds=%5B%22"
        template_sep = "%22%2C%22"
        template_end = "%22%5D"
        raw_image_dir = self._raw_dir / "images"
        raw_image_dir.mkdir(exist_ok=True)
        image_ids = list(metadata_df.index)
        with tqdm(
            total=(len(image_ids) - 1) // 50 + 1, desc="Downloading images", colour=self._pbar_col
        ) as pbar:
            for i, block in enumerate(self.chunk(image_ids, 50)):
                pbar.set_postfix(image_id=block[0])
                args = ""
                args += template_start
                args += template_sep.join(block)
                args += template_end
                req = requests.get(f"{self._rest_api_url}/image/download{args}", stream=True)
                req.raise_for_status()
                image_path = raw_image_dir / f"{i}.zip"
                with open(image_path, "wb") as f:
                    shutil.copyfileobj(req.raw, f)
                del req
                pbar.update()

    def _preprocess_isic_metadata(self) -> None:
        """Preprocesses the raw ISIC metadata."""
        self._processed_dir.mkdir(exist_ok=True)

        metadata_path = self._raw_dir / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "metadata.csv not found while preprocessing ISIC dataset. "
                "Run `download_isic_metadata` and `download_isic_images` before "
                "calling `preprocess_isic_metadata`."
            )
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df.set_index("_id")
        labels_df = self._remove_uncertain_diagnoses(metadata_df)
        labels_df = self._add_patch_column(labels_df)

        labels_df["path"] = (
            str(self._processed_dir)  # type: ignore
            + os.sep
            + "ISIC-images"
            + os.sep
            + labels_df["dataset.name"]
            + os.sep
            + labels_df["name"]
            + ".jpg"
        )
        labels_df.to_csv(self._processed_dir / "labels.csv")

    def _preprocess_isic_images(self) -> None:
        """Preprocesses the images."""
        if (self._processed_dir / "ISIC-images").is_dir():
            return
        if not (self._raw_dir / "images").is_dir():
            raise FileNotFoundError(
                "Raw ISIC images not found. Run `download_isic_images` before "
                "calling `preprocess_isic_images`."
            )
        labels_df = pd.read_csv(self._processed_dir / "labels.csv")
        labels_df = labels_df.set_index("_id")
        image_ids = labels_df.index.tolist()

        self._processed_dir.mkdir(exist_ok=True)
        image_zips = tuple((self._raw_dir / "images").glob("**/*.zip"))
        with tqdm(total=len(image_zips), desc="Unzipping images", colour=self._pbar_col) as pbar:
            for file in image_zips:
                pbar.set_postfix(file_index=file.stem)
                with zipfile.ZipFile(file, "r") as zip_ref:
                    zip_ref.extractall(self._processed_dir)
                    pbar.update()
        images = []
        for ext in ("jpg", "jpeg", "png"):
            images.extend(self._processed_dir.glob(f"**/*.{ext}"))
        with tqdm(total=len(images), desc="Processing images", colour=self._pbar_col) as pbar:
            for image_path in images:
                pbar.set_postfix(image_name=image_path.stem)
                image = Image.open(image_path)
                image = image.resize((224, 224))  # Resize the images to be of size 224 x 224
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                image.save(image_path.rename(image_path.with_suffix(".jpg")))
                pbar.update()

    @staticmethod
    def _remove_uncertain_diagnoses(metadata_df: pd.DataFrame) -> pd.DataFrame:
        labels_df = metadata_df.loc[
            metadata_df["meta.clinical.benign_malignant"].isin({"benign", "malignant"})
        ]  # throw out unknowns
        malignant_mask = labels_df["meta.clinical.benign_malignant"] == "malignant"
        labels_df["malignant"] = malignant_mask.astype(np.uint8)

        labels_df["meta.clinical.diagnosis_confirm_type"].fillna(
            value="non-histopathology", inplace=True
        )
        histopathology_mask = labels_df["meta.clinical.diagnosis_confirm_type"] == "histopathology"
        labels_df["histo"] = histopathology_mask.astype(np.uint8)

        return labels_df

    @staticmethod
    def _add_patch_column(labels_df: pd.DataFrame) -> pd.DataFrame:
        """Adds a patch column to the input DataFrame."""
        patch_mask = labels_df["dataset.name"] == "SONIC"
        # add to labels_df
        labels_df["patch"] = None
        labels_df.loc[patch_mask, "patch"] = 1
        labels_df.loc[~patch_mask, "patch"] = 0
        assert not any(patch is None for patch in labels_df["patch"])
        return labels_df

    def _download_data(self) -> None:
        """Attempt to download data if files cannot be found in the base folder."""
        # # Check whether the data has already been downloaded - if it has and the integrity
        # # of the files can be confirmed, then we are done
        if self._check_downloaded():
            LOGGER.info("Files already downloaded and verified.")
            return
        # Create the directory and any required ancestors if not already existent
        self._data_dir.mkdir(exist_ok=True, parents=True)
        LOGGER.info(f"Downloading metadata into {str(self._raw_dir / 'metadata.csv')}...")
        self._download_isic_metadata()
        LOGGER.info(
            f"Downloading data into {str(self._raw_dir)} for up to {self.max_samples} samples..."
        )
        self._download_isic_images()

    def _preprocess_data(self) -> None:
        """Preprocess the downloaded data if the processed image-directory/metadata don't exist."""
        # If the data has already been processed, skip this operation
        if self._check_processed():
            LOGGER.info("Metadata and images already preprocessed.")
            return
        LOGGER.info(
            f"Preprocessing metadata (adding columns, removing uncertain diagnoses) and saving into "
            f"{str(self._processed_dir / 'labels.csv')}..."
        )
        self._preprocess_isic_metadata()
        LOGGER.info(
            f"Preprocessing images (transforming to 3-channel RGB, resizing to 224x224) and saving "
            f"into{str(self._processed_dir / 'ISIC-images')}..."
        )
        self._preprocess_isic_images()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Sample:
        image = Image.open(self.x[index])
        if self.transform is not None:
            image = self.transform(image)
        target = self.y[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return Sample(x=image, s=self.s[index], y=target)
