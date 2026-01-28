"""
Object-Centric Semantic Scoring for VLFM

This module contains our capstone implementation for improving VLFM with:
- Object-centric semantic scoring (vs. frontier-based)
- Mobile-SAM for object segmentation
- SigLIP for vision-language features
- HOV-SG-style weighted feature fusion
- Persistent 3D object mapping (ConceptGraphs-inspired)
"""

from .sam_detector import SAMDetector
from .siglip2 import SigLIP
from .object_detection import ObjectSegmenter, Detection

__all__ = [
    "SAMDetector",
    "SigLIP",
    "ObjectSegmenter",
    "Detection",
]
