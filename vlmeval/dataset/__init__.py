"""Only the image datasets supported by this release."""
from .image_mcq import ImageMCQDataset, MMMUDataset
from .image_yorn import ImageYORNDataset

SUPPORTED_DATASETS = ['HallusionBench', 'MMMU_DEV_VAL', 'MMStar', 'POPE']


def build_dataset(dataset_name, **kwargs):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    cls = MMMUDataset if dataset_name == 'MMMU_DEV_VAL' else (
        ImageYORNDataset if dataset_name in ('HallusionBench', 'POPE') else ImageMCQDataset)
    return cls(dataset=dataset_name, **kwargs)


def DATASET_TYPE(dataset, *, default='MCQ'):
    return 'Y/N' if dataset in ('HallusionBench', 'POPE') else default


def DATASET_MODALITY(dataset, *, default='IMAGE'):
    return default

