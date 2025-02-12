import torch
import numpy as np
import cv2
from PIL import Image
from config import *
from pathlib import Path

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def initialize_sam_model(checkpoint_path = Path(SAM_CHECKPOINT), model_type = "vit_h", device = DEVICE):
    # print(checkpoint_path, type(checkpoint_path))
    ''' Load the saved weights and init the sam model '''
    sam_model = sam_model_registry[ model_type ](checkpoint = checkpoint_path)
    sam_model = sam_model.to(DEVICE)
    return sam_model


def preprocess():
    ''' 1. Crops and makes a square image
        2. Resize image to 512, 512 (for memory issues)
    '''
    source_image = Image.open(INPUT_IMAGE_PATH)
    target_height, target_width = 512, 512
    width, height = source_image.size
    source_image = source_image.crop((0, height-width, width, height)) # bbox: [left, up, right, down]

    # resize
    source_image = source_image.resize( size = (target_width, target_height), 
                        resample = Image.LANCZOS )
    
    # numpify
    segmented_image = np.asarray(source_image)
    return source_image, segmented_image


def generate_masks(sam_model, image, pred_iou_thresh=0.99, points_per_side = 32):
    ''' Generate masks for a given image and return the masks'''
    mask_generator = SamAutomaticMaskGenerator( pred_iou_thresh = pred_iou_thresh, 
                                                points_per_side = points_per_side, 
                                                model = sam_model,
                                                stability_score_thresh = 0.92,
                                                min_mask_region_area = 100,
                                                crop_n_layers=1,
                                                crop_n_points_downscale_factor=2,
                                                )
    return mask_generator.generate(image)






if __name__ == "__main__":
    sam_model = initialize_sam_model()
    print( type(sam_model) )

    ### Example image from unsplash.com
    ### Photo by Lac McGregor, Canada
    ### Free to use under the Unsplash License
    ### Link: https://unsplash.com/photos/AsJirOOLN_s
    # Get the image from input_images

    source_image, segmented_image = preprocess()
    masks = generate_masks(sam_model, segmented_image)

    print( type(masks) )
    print( masks )





