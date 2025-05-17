"""
Extraction module package for removing backgrounds from images using various methods.
"""

from .custom_extraction import extract_object_from_photo, batch_extract_objects, enhance_extracted_object
from .rembg_extraction import remove_background_with_rembg, remove_background_with_rembg_simple, batch_remove_background_with_rembg, batch_process_cartier_with_birefnet
from .pixellib_extraction import remove_background_with_pixellib, remove_background_with_pixellib_simple
from .segment_anything_extraction import remove_background_with_segment_anything, remove_background_with_segment_anything_simple
from .dis_bg_extraction import remove_background_with_dis_simple
from .background_remover_tool import remove_background_br

__all__ = [
    'extract_object_from_photo', 
    'batch_extract_objects', 
    'enhance_extracted_object',
    'remove_background_with_rembg',
    'remove_background_with_rembg_simple',
    'batch_remove_background_with_rembg',
    'batch_process_cartier_with_birefnet',
    'remove_background_with_pixellib',
    'remove_background_with_pixellib_simple',
    'remove_background_with_segment_anything',
    'remove_background_with_segment_anything_simple',
    # 'remove_background_with_dis',
    'remove_background_with_dis_simple',
    'remove_background_br'
]