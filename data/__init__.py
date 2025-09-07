'''
Data module for Nova Score.

This module contains data loading, generation, and preprocessing utilities.
'''

from .simulate import generate_synthetic_partners, save_synthetic_data, generate_synthetic_data

__all__ = [
    'generate_synthetic_partners',
    'save_synthetic_data',
    'generate_synthetic_data'
]
