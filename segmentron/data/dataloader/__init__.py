"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .cityscapes import CitySegmentation
from .densepass import DensePASSSegmentation

datasets = {
    'cityscape': CitySegmentation,
    'densepass': DensePASSSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
