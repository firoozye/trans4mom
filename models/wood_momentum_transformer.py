"""
Backwards-compatibility facade for WoodMomentumTransformer.
The canonical implementation is in models/momentum_transformer.py.
"""
from models.momentum_transformer import WoodMomentumTransformer, MomentumTransformer

__all__ = ['WoodMomentumTransformer', 'MomentumTransformer']
