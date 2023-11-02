#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python
import os 
import multiprocessing
from tqdm import tqdm

# Images
import cv2
from PIL import Image
import PIL.ImageOps 

# Mathematics
import numpy as np
import math

# Modules
from mol_depict.utils.utils_image import resize_image


def get_ocr(force_cpu):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls = True, 
        lang = 'en', 
        det_db_thresh = 0.5, 
        det_db_box_thresh = 0.5, 
        det_db_unclip_ratio = 1.5, 
        use_dilation = False, 
        det_db_score_mode = 'fast', 
        drop_score = 0.5,
        det_limit_side_len = 2500,
        max_batch_size = 50,
        show_log = False,
        det_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_det_infer/",
        rec_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_rec_infer/",
        cls_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/ch_ppocr_mobile_v2.0_cls_infer",
        use_gpu = not(force_cpu),
    )  
    return ocr 

class CaptionRemover:
    def __init__(self, config = None, force_cpu = False, remove_captions=True):
        self.config = config
        self.border_size = 30
        self.ocr = get_ocr(force_cpu = force_cpu)
        self.force_cpu = force_cpu
        self.remove_captions = remove_captions

    def _preprocess_images_process(self, images_filenames):
        pil_images = []
        for image_filename in tqdm(images_filenames):
            pil_image = Image.open(image_filename)
                
            pil_image = resize_image(
                pil_image, 
                (self.config["image_size"][1], self.config["image_size"][1]), 
                border_size = self.border_size
            )
            
            if self.remove_captions:
                # Remove captions (95% of the dataloading time)
                image = self(pil_image)
                pil_image = Image.fromarray(image).convert('RGB')

            # Remove borders
            pil_image = crop_tight(pil_image)
            
            # Resize, add small borders 
            pil_image = resize_image(
                pil_image,
                (self.config["image_size"][1], self.config["image_size"][1]), 
                border_size = self.border_size*3 
            )
            pil_images.append(pil_image)
        return pil_images

    def preprocess_images(self, images_filenames):
        return self._preprocess_images_process(images_filenames)
   
    def __call__(self, pil_image):
        """
        This function removes the caption in a chemical structure image.
        Args:
            image_path (str): the path to locate the image that needs to be cleaned.
        Returns:
            image: Clean image.
        """
        image = np.array(pil_image,  dtype=np.uint8)
        image = np.stack((image, )*3, axis=-1)
        self.image = image
            
        # Run OCR to determine whether or not to apply the mask filter
        result = self.ocr.ocr(self.image)
        if result == [None]:
            return self.image
        self.boxes = [line[0] for line in result[0]]
        self.texts = [line[1][0] for line in result[0]]
        self.scores = [line[1][1] for line in result[0]]
        # Get indices numbers to remove if there is any prior filter by OCR:
        filter_masks, remove_indices = self.detect_strange_captions()
        
        # Remove boxes
        if len(remove_indices) != 0:
            remove_boxes = [self.boxes[i] for i in range(len(self.boxes)) if i in remove_indices]
            for box in remove_boxes:
                box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
                cv2.fillPoly(self.image, [box], color=(255, 255, 255))
        
        # Apply mask filter
        if filter_masks:
            masks = self.generate_masks()
            self.image = self.remove_smaller_masks(masks)
        
        return self.image

    def detect_strange_captions(self):
        """
        This function determines (1) if any of the boxes detected by the OCR should be
        removed, and (2) if it is needed to run a mask filtering.
        Returns:
            filter_masks (bool): Returns whether or not we need to filter by mask size.
            remove_indices (list): Returns a list with the indices of the boxes to remove by OCR.
        """
        characters_allowed = ['C', 'c', 'N', 'H', 'O', 'o', 'h', 'n']
        filter_masks = False
        remove_indices = []
        for i in range(len(self.texts)):
            number_of_characters_allowed = len([substring for substring in self.texts[i] if substring in characters_allowed])
            if (len(self.texts[i]) > 10):
                filter_masks = True
                if (number_of_characters_allowed < len(self.texts[i])/3):
                    remove_indices.append(i)
            elif (len(self.texts[i]) > 2) and (number_of_characters_allowed < len(self.texts[i])/2):
                filter_masks = True
            elif (self.texts[i].isdigit()) and (int(self.texts[i]) != 0) and (self.scores[i] >= 0.85):
                filter_masks = True
            elif (len(self.texts[i]) > 2) and (self.texts[i].count(',') > 2):
                filter_masks = True
            elif (len(self.texts[i]) == 2) and any(substring in self.texts[i] for substring in ['(', ')', ':']):
                filter_masks = True

        return filter_masks, remove_indices

    def remove_smaller_masks(self, mask):
        """
        This function keeps only the biggest mask on the image which should be
        the molecule.
        Args:
            mask (numpy array): Masked image
        Returns:
            image (numpy array): Image filtered where only the biggest mask is 
            retained.
        """
        contours, _ = cv2.findContours(mask[:,:,0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_size = [len(cnt) for cnt in contours]
        if len(contours_size) > 0:
            max_index = contours_size.index(max(contours_size))
        else:
            return self.image

        for i in range(len(contours)):
            if i != max_index:
                epsilon = 0.01 * cv2.arcLength(contours[i], True)
                approx = cv2.approxPolyDP(contours[i], epsilon, True)
                self.image = cv2.drawContours(self.image, [approx], 0, (255,255,255), thickness=cv2.FILLED)
            
        return self.image

    def generate_masks(self):
        """
        This function returns a masked image where the molecule and captions are
        hopefully separated.
        Args:
            image (numpy array): image to process and convert into masked image.
        Returns:
            dilated_mask (numpy array): masked image.
        """
        mask = cv2.threshold(self.image, 254, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_not(mask)
        dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=2)
        flood_fill_margin = 0
        dilated_mask = cv2.copyMakeBorder(dilated_mask, flood_fill_margin, flood_fill_margin, flood_fill_margin, flood_fill_margin, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        flood_fill_mask = np.zeros((dilated_mask.shape[0] + 2, dilated_mask.shape[1] + 2), np.uint8)
        cv2.floodFill(dilated_mask, flood_fill_mask, (0,0), 255)  
        is_black_pixel = np.where((dilated_mask[:, :, 0] == 0) & (dilated_mask[:, :, 1] == 0) & (dilated_mask[:, :, 2] == 0))
        dilated_mask[is_black_pixel] = [255, 255, 255]
        is_black_pixel = np.where((dilated_mask[:, :, 0] == 255) & (dilated_mask[:, :, 1] == 0) & (dilated_mask[:, :, 2] == 0))
        dilated_mask[is_black_pixel] = [0, 0, 0]     
        return dilated_mask

def get_bond_size(keypoints):
    min_distances = []
    for index_keypoint_query, keypoint_query in enumerate(keypoints):
        distances = []
        for index_keypoint_key, keypoint_key in enumerate(keypoints):
            if index_keypoint_key < index_keypoint_query:
                distance = math.dist(keypoint_query, keypoint_key)
                distances.append(distance)

        if len(distances) > 0:
            min_distances.append(min(distances))

    if len(min_distances) > 0:
        # Upper bound estimate of the bond length, as the 75th percentile of minimal distances
        bond_length = np.percentile(min_distances, 75) 
    else:
        print("Get bond size error")
        bond_length = 100
    return bond_length

def get_bonds_sizes(keypoints_list, scaling_factor):
    bonds_sizes = []
    for keypoints in keypoints_list:
        keypoints = [
            [(keypoint[0]*scaling_factor + scaling_factor//2), 
            (keypoint[1]*scaling_factor + scaling_factor//2)] for keypoint in keypoints
        ]
        bonds_sizes.append(get_bond_size(keypoints))
    return bonds_sizes

def crop_tight(pil_image, dilate=False):
    if dilate:
        # Dilate image to remove isolated pixels
        im = np.array(pil_image, dtype=np.uint8)
        im = cv2.dilate(im, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        im = Image.fromarray(im)
    else:
        im = pil_image.copy()

    bbox = PIL.ImageOps.invert(im).getbbox()
    if bbox is None:
        return pil_image
    
    min_cropping_size = 60
    if (abs(bbox[2] - bbox[0]) > min_cropping_size) or (abs(bbox[3] - bbox[1]) > min_cropping_size):
        return pil_image.crop(bbox)

    elif dilate:
        print("Recursive borders cropping")
        return crop_tight(pil_image, dilate=False)

    else:
        print("Fixed window cropping")
        missing_x = min_cropping_size - abs(bbox[2] - bbox[0]) 
        missing_y = min_cropping_size - abs(bbox[3] - bbox[1])
        print(bbox)
        bbox = [
            bbox[0] - missing_x//2, 
            bbox[1] - missing_y//2, 
            bbox[2] + missing_x//2, 
            bbox[3] + missing_y//2
        ]
        return pil_image.crop(bbox)
    
