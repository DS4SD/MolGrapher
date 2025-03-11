#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mathematics
import numpy as np

# Pytorch 
import torch 

# Image
import cv2
from PIL import Image

# OCR
from paddleocr import PaddleOCR
from weighted_levenshtein import lev

# Python
import multiprocessing
from tqdm import tqdm
import os
import json 

# Modules
from molgrapher.utils.utils_dataset import CaptionRemover, crop_tight
from mol_depict.utils.utils_image import resize_image


def get_ocr_recognition_only(force_cpu = False):
    ocr_recognition_only = PaddleOCR(
        lang='en', 
        drop_score = 1e-20, 
        image_orientation = True, 
        rec_image_inverse = False, 
        rec_algorithm = 'CRNN', 
        det = False, 
        #rec_image_shape = [1,50,50], 
        use_angle_cls = True, 
        max_text_length = 60,
        show_log = False,
        det_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_det_infer/",
        rec_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_rec_infer/",
        cls_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/ch_ppocr_mobile_v2.0_cls_infer",
        use_gpu = not(force_cpu)
    )
    return ocr_recognition_only 

def get_ocr(force_cpu = False):
    ocr = PaddleOCR(
        use_angle_cls = True, 
        lang = 'en', 
        det_db_thresh = 1e-8,
        det_db_box_thresh = 0, 
        det_db_unclip_ratio = 3, 
        use_dilation = True, 
        det_db_score_mode = 'slow', 
        drop_score = 1e-15,
        show_log = False,
        det_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_det_infer/",
        rec_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_rec_infer/",
        cls_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/ch_ppocr_mobile_v2.0_cls_infer",
        use_gpu = not(force_cpu)
    )
    return ocr 

def get_ocr_angle(force_cpu):
    ocr_angle = PaddleOCR(
        use_angle_cls = True, 
        show_log = False,
        rec_image_inverse = False, 
        det_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_det_infer/",
        rec_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_rec_infer/",
        cls_model_dir = os.path.dirname(__file__) + "/../../data/external/paddleocr/ch_ppocr_mobile_v2.0_cls_infer",
        use_gpu = not(force_cpu)
    )
    return ocr_angle


class AbbreviationDetector:
    def __init__(self, config, image_size = (1024, 1024), force_cpu = False, force_no_multiprocessing = False, angle_recognition = False):
        self.image_size = image_size
        self.config = config
        self.border_size = 30
        self.force_cpu = force_cpu
        self.force_no_multiprocessing = force_no_multiprocessing  
        self.angle_recognition = angle_recognition
        # Set the number of threads used in pytorch to 1 to avoid conflicts with MKL.
        torch.set_num_threads(1)

    def _detect_process(self, images_filenames_list, bonds_sizes_list):
        abbreviations_list = []
        for image_filename, bond_size in tqdm(zip(images_filenames_list, bonds_sizes_list), total=len(bonds_sizes_list)):
            pil_image = Image.open(image_filename)
            pil_image = resize_image(
                pil_image, 
                (self.config["image_size"][1], self.config["image_size"][1]), 
                border_size = self.border_size
            )

            # Remove captions
            image = self.caption_remover(pil_image)
            
            # Remove borders
            pil_image = Image.fromarray(image).convert('RGB')
            pil_image = crop_tight(pil_image)
            
            # Resize, add small borders 
            pil_image = resize_image(
                pil_image,
                (self.config["image_size"][1], self.config["image_size"][1]), 
                border_size = self.border_size*3 
            )
            
            # Threshold and convert to float
            image = np.array(pil_image, dtype=np.float32)/255
            image[image > 0.6] = 1. 
            image[image != 1.] = 0.
            image = np.stack((image, )*3, axis=-1)

            image = np.array(image*255, dtype=np.uint8)
            abbreviations_list.append(self(image, bond_size))
            
        return abbreviations_list

    def _detect_process_star(self, args):
        return self._detect_process(*args)

    def mp_run(self, images_filenames, graphs, bonds_sizes, filter=False):
        if filter:
            # Filter out graphs which do not contain any abbreviations
            images_filenames_abbreviations = []
            bonds_sizes_abbreviations = []
            for image_filename, bond_size, graph in zip(images_filenames, bonds_sizes, graphs):
                if graph.needs_abbreviations_detection():
                    images_filenames_abbreviations.append(image_filename)
                    bonds_sizes_abbreviations .append(bond_size)
            images_filenames = images_filenames_abbreviations
            bonds_sizes = bonds_sizes_abbreviations 
        
        if self.force_cpu and not(self.force_no_multiprocessing):
            # Debugging 
            #print(self._detect_process(images_filenames, bonds_sizes, self.ocr_recognition_only, self.ocr))

            if len(images_filenames) < self.config["num_processes_mp"]:
                print("Abbreviation detector warning: Too much processes")
            images_filenames_split = np.array_split(images_filenames, self.config["num_processes_mp"])
            bonds_sizes_split = np.array_split(bonds_sizes, self.config["num_processes_mp"])
            args = [[images_filenames_split[process_index], bonds_sizes_split[process_index]] for process_index in range(self.config["num_processes_mp"])]
            pool = multiprocessing.Pool(self.config["num_processes_mp"])
            abbreviations_processes = pool.map(self._detect_process_star, args)
            pool.close()
            pool.join()

            abbreviations_list = []
            for index in range(self.config["num_processes_mp"]):
                abbreviations_list.extend(abbreviations_processes[index])
        else:
            abbreviations_list = self._detect_process(images_filenames, bonds_sizes)
        
        # Add empty lists for images which do not contain any abbreviations
        i = 0
        abbreviations_list_return = []
        for graph in graphs:
            if filter:
                if graph.needs_abbreviations_detection():
                    abbreviations_list_return.append(abbreviations_list[i])
                    i += 1
                else:
                    abbreviations_list_return.append([])
            else:
                abbreviations_list_return.append(abbreviations_list[i])
                i += 1

        return abbreviations_list_return

    def __call__(self, image, bond_size = 100, image_filename = None):
        self.image = image
        self.original_image = image
        self.bond_size = bond_size
        self.image_filename = image_filename
        
        # Get image containing selected regions (corresponding to individual atom letters)
        self.filtered_image = self.filter_image(self.image, bond_size)

        # Set atom label crops and boxes (by merging individual atom letters)
        self.set_label_crops_and_boxes()

        # Predict
        image_org, texts, scores, boxes = self.get_label_crops_predictions()
        abbreviations = [{
            "text": text,
            "box": [box[0], box[2]],
            "score": score,
        } for text, box, score in zip(texts, boxes, scores)]
            
        return abbreviations
        
    def filter_image(self, image, bond_size = 100, pr_mode = 3, 
                kernel_size = 2, contour_method = cv2.RETR_EXTERNAL, 
                contour_approx_mode = cv2.CHAIN_APPROX_SIMPLE):
    
        """
        This function cleans an image returning only the contours that represent label crops.
        The code has been modified from: 
        https://www.kaggle.com/code/thomaskonstantin/letter-retrieval-molecular-translation/notebook
        
        Args:
            img (np.array): an image of a chemical compound.
            pr_mode (int): preprocessing preset value can be 1,2 or 3 
            kernel_size (int): kernel size for morphological operations in preprocessing  
        """
        img = image[:, :, 0]
        t1 = img.copy().astype(np.uint8)
        
        # Preprocess the image so better contours can be detected
        if pr_mode == 1:
            t1 = cv2.erode(t1, np.ones((kernel_size, kernel_size)))
        elif pr_mode == 2:
            t1 = cv2.morphologyEx(t1, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size)))
        else:
            t1 = cv2.morphologyEx(t1, cv2.MORPH_GRADIENT, np.ones((kernel_size, kernel_size)))

        img = cv2.cvtColor(np.zeros_like(t1), cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(t1.astype(np.uint8), contour_method, contour_approx_mode)
        cnts = self.get_contours(contours, bond_size/1.5, bond_size*75, bond_size/2.5) 
        wc = cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
        wc =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return cv2.bitwise_and(t1, t1, mask = wc)

    def get_contours(self, contours, beta1, beta2, alpha, beta=0.2): #0.2
        """
        This function filters the contours found on the image based on the minimum and maximum
        countour area (beta1 and beta2) and the difference of the bounding box sizes (alpha).

        Args:
            contours (cv2 contours): arrays with the contours of the image.
            alpha (float): the maximum allowed difference between bouding rectangle sides, 0 difference means all sides are equal i.e square.
            beta1 (float): the minimum area of the circle which bound the contour found.
            beta2 (int): the maximum area of the circle which bound the contour found.
        
        Returns:
            cnts (cv2 contours): arrays with the contours of the image filtered.
        """
        cnts = []
        bsd = lambda x: np.abs(np.linalg.norm(x[0] - x[1]) - np.linalg.norm(x[0] - x[3]))
        for cnt in contours:
            _,radius = cv2.minEnclosingCircle(cnt)
            rect = cv2.minAreaRect(cnt) # (center (x,y), (width, height), angle of rotation)
            box = cv2.boxPoints(rect)
            area = np.pi * radius**2
            if (beta1 > area ) or (area > beta2) or (bsd(box) > alpha):
                continue 
            if not(rect[2] in [0, -90, 90, 180, -180, 270, -270, 360, -360]) and (min(rect[1][0],rect[1][1])/max(rect[1][0],rect[1][1]) < beta):
                continue
            cnts.append(cnt)      
        return cnts
    
    def set_label_crops_and_boxes(self, scope = 1.05, contour_method = cv2.RETR_EXTERNAL, contour_approx_mode = cv2.CHAIN_APPROX_SIMPLE):
        """
        This function gets the label boxes, expands them based on distance
        to other labels and generates new bounding boxes based on that expansion.
        Args:
            scope (int, optional): The scope is used to expand the bounding box to retrieve the individual arrays. 
                                    Defaults to 1.
            contour_method (cv2 method, optional): Method to retrieve the contour: . Defaults to cv2.RETR_EXTERNAL.
            contour_approx_mode (cv2 method, optional): Defaults to cv2.CHAIN_APPROX_SIMPLE.
        """
        contours, _ = cv2.findContours(self.filtered_image.astype(np.uint8), contour_method, contour_approx_mode)

        boxes = []
        label_crops = []
        
        # Get initial boxes
        for cts in contours:
            rect = cv2.minAreaRect(cts)
            box = np.int0(cv2.boxPoints(rect))
            min_x, max_x, min_y, max_y = min([i[0] for i in box]), max([i[0] for i in box]), min([i[1] for i in box]), max([i[1] for i in box])
            boxes.append([[min_x, min_y], [max_x, max_y]])

        # Merge neighboring boxes (each letter of an atom label)
        boxes = self.get_overlapping_boxes(boxes)
        
        # Get updated boxes and label crops images
        new_boxes = []
        for box in boxes:
            min_x, max_x, min_y, max_y = min([i[0] for i in box]), max([i[0] for i in box]), min([i[1] for i in box]), max([i[1] for i in box])
            center = (int((min_x + max_x)/2), int((min_y + max_y)/2))
            size = (int(scope*(max_x - min_x)), int(scope*(max_y - min_y)))
            cropped = cv2.getRectSubPix(self.image, size, center)/255
            cropped[cropped < 0.9] = 0
            cropped[cropped != 0] = 1
            label_crops.append(cropped)
            new_boxes.append(
                np.array([
                    [box[0][0] + 1, box[0][1] - 1],
                    [box[1][0] + 1, box[0][1] - 1],
                    [box[1][0] + 1, box[1][1] + 1],
                    [box[0][0] - 1, box[1][1] + 1]
                ])
            )
        self.clean_image(label_crops, new_boxes)
        self.label_crops = label_crops
        self.boxes = new_boxes
    
    def get_overlaps(self, boxes, anchor_box, index):
        overlaps = []
        for a in range(len(boxes)):
            if a == index:
                continue
            if (boxes[a][0][0] >= anchor_box[1][0])  or \
                (boxes[a][1][0] <= anchor_box[0][0]) or \
                (boxes[a][0][1] >= anchor_box[1][1]) or \
                (boxes[a][1][1] <= anchor_box[0][1]):
                continue
            overlaps.append(a)
        return overlaps

    def get_overlapping_boxes(self, boxes, merge_margin=30):
        """
        This function expands the individual character boxes. 
        Args:
            boxes (list of ararys): list with the individual character bounding boxes.
        Returns:
            boxes (list of ararys): list with the expanded bounding boxes.
        """
        finished = False
        while not(finished):
            finished = True
            index = len(boxes) - 1
            while index >= 0:
                # Add margin
                top_left = boxes[index][0][:]
                bottom_right = boxes[index][1][:]
                top_left[0] -= merge_margin
                #top_left[1] -= merge_margin
                bottom_right[0] += merge_margin
                #bottom_right[1] += merge_margin
                # Get matching boxes
                overlaps = self.get_overlaps(boxes, [top_left, bottom_right], index)
                if len(overlaps) == 0:
                    index -= 1
                    continue
                # Combine boxes
                contours = []
                overlaps.append(index)
                for i in overlaps:
                    contours.append([boxes[i][0]])
                    contours.append([boxes[i][1]])
                # Get bounding rect
                x, y, w, h = cv2.boundingRect(np.array(contours))
                w -= 1
                h -= 1
                merged = [[x, y], [x + w, y + h]]
                # Remove boxes from list
                overlaps.sort(reverse = True)
                boxes = [b for i, b in enumerate(boxes) if i not in overlaps]
                boxes.append(merged)
                finished = False
                break
        return boxes
    
    def clean_image(self, label_crops, boxes):
        """
        This function cleans the original image leaving only the parts from the expanded
        bounding boxes detected.
        """
        # Initialize empty mask
        mask = np.zeros(self.image_size, dtype=np.uint8)+1 

        # Add the bounding boxes filtered from the original image to the mask
        for i in range(len(boxes)):
            minx = min(boxes[i][:, 1])
            miny = min(boxes[i][:, 0])
            try:
                mask[
                    minx : minx + label_crops[i].shape[0], 
                    miny : miny + label_crops[i].shape[1]
                ] = label_crops[i][:, :, 0]
            except:
                print("Warning: label crop setting incorrect.")
                pass
        mask = mask*255
        self.image = np.stack((mask, )*3, axis=-1)
     
    def get_label_crops_predictions(self, display_paddle_predictions=False, display_paddle_predictions_individual=False):
        """
        Returns:
            image (np.array): original image.
            texts (list of str): list with the predicted texts.
            scores (list of float): list with the prediction scores.
            boxes (list of arrays): list with the bounding boxes.
        """
        texts, scores, new_boxes = [], [], []
        for i in range(len(self.boxes)):  

            # If the label crop height is smaller than a given threshold
            if (self.label_crops[i].shape[0] < self.bond_size/1.5):# and (self.label_crops[i].shape[0] < 0.4*self.bond_size):
                # Recognition only for small boxes
                predictions = self.ocr_recognition_only.ocr(self.label_crops[i]*255, det = False, rec = True, cls = False)
                texts.append(predictions[0][0][0])
                scores.append(predictions[0][0][1])
                new_boxes.append(self.boxes[i])
            else:                   
                # Detection + Recognition 
                predictions = self.ocr.ocr(self.label_crops[i]*255, det = True, rec = True, cls = False)[0]  
                if not(predictions):
                    # Fallback option
                    predictions = self.ocr_recognition_only.ocr(self.label_crops[i]*255, det = False, rec = True, cls = False)
                    texts.append(predictions[0][0][0])
                    scores.append(predictions[0][0][1])
                    new_boxes.append(self.boxes[i])
                    continue
               
                label_crop_texts, label_crop_scores, label_crop_boxes = [], [], []
                for prediction in predictions:
                    label_crop_texts.append(prediction[1][0])
                    label_crop_scores.append(prediction[1][1])
                    # Translate Paddle OCR boxes to image reference
                    label_crop_boxes.append(
                        np.array([
                            [prediction[0][0][0] + self.boxes[i][0][0], prediction[0][0][1] + self.boxes[i][0][1]],
                            [prediction[0][0][0] + self.boxes[i][1][0], prediction[0][0][1] + self.boxes[i][1][1]],
                            [prediction[0][0][0] + self.boxes[i][2][0], prediction[0][0][1] + self.boxes[i][2][1]],
                            [prediction[0][0][0] + self.boxes[i][3][0], prediction[0][0][1] + self.boxes[i][3][1]]
                        ])
                    )
                if "S" in label_crop_texts and (("O2" in label_crop_texts) or ("02" in label_crop_texts) or ("Oz" in label_crop_texts) or ("0z" in label_crop_texts)):
                    if not(all(label_crop_text in ["S", "O2", "02", "Oz", "0z"]) for label_crop_text in label_crop_texts):
                        continue 
                    label_crop_texts = ["SO2"]
                    label_crop_scores = [1]
                    label_crop_boxes = [self.boxes[i]]
                if "C" in label_crop_texts and (("F2" in label_crop_texts) or ("Fz" in label_crop_texts)):
                    if not(all(label_crop_text in ["C", "F2", "Fz"]) for label_crop_text in label_crop_texts):
                        continue 
                    label_crop_texts = ["CF2"]
                    label_crop_scores = [1]
                    label_crop_boxes = [self.boxes[i]]
                texts.extend(label_crop_texts)
                scores.extend(label_crop_scores)
                new_boxes.extend(label_crop_boxes)
            
        return self.image, texts, scores, new_boxes 


class AbbreviationDetectorGPU(AbbreviationDetector):
    def __init__(self, config, image_size = (1024, 1024), force_cpu = False, force_no_multiprocessing = False, angle_recognition = False):
        super(AbbreviationDetectorGPU, self).__init__(config, image_size, force_cpu, force_no_multiprocessing, angle_recognition)
        self.caption_remover = CaptionRemover(force_cpu = False)
        self.ocr = get_ocr(force_cpu = False)
        self.ocr_recognition_only = get_ocr_recognition_only(force_cpu = False)
        if angle_recognition:
            self.ocr_angle = get_ocr_angle(force_cpu = False)


class AbbreviationDetectorCPU(AbbreviationDetector):
    caption_remover = CaptionRemover(force_cpu = True)
    ocr = get_ocr(force_cpu = True)
    ocr_recognition_only = get_ocr_recognition_only(force_cpu = True)
    
    def __init__(self, config, image_size = (1024, 1024), force_cpu = True, force_no_multiprocessing = False, angle_recognition = False):
        super(AbbreviationDetectorCPU, self).__init__(config, image_size, force_cpu, force_no_multiprocessing, angle_recognition)


class SpellingCorrector:
    def __init__(self, abbreviations_smiles_mapping=None):
        self.abbreviations = abbreviations_smiles_mapping
        with open(os.path.dirname(__file__) + "/../../data/ocr_mapping/ocr_atoms_classes_mapping.json") as file:
            ocr_atoms_classes_mapping = json.load(file)
        self.abbreviations = {**self.abbreviations, **ocr_atoms_classes_mapping} 
        self.set_costs()
        
    def __call__(self, predicted_abbreviation):       
        if predicted_abbreviation in self.abbreviations:
            return predicted_abbreviation

        score_min = 150
        indices_to_check = []
        for i, correct_abb in enumerate(self.abbreviations):
            levenshtein_score = lev(
                predicted_abbreviation, 
                correct_abb, 
                substitute_costs = self.substitute_costs, 
                insert_costs = self.insert_costs, 
                delete_costs = self.delete_costs
            )
            if levenshtein_score < score_min:
                score_min = levenshtein_score
                indices_to_check = [i]
            elif levenshtein_score == score_min:
                indices_to_check.append(i)

        if len(indices_to_check) == 1:
            return list(self.abbreviations.keys())[indices_to_check[0]]
        else:
            # Select the most popular abbreviation
            keys_candidates = [list(self.abbreviations.keys())[index] for index in indices_to_check]
            abbreviations_populations = {k: v["population"] for k, v in self.abbreviations.items() if k in keys_candidates}
            max_population = 0
            for abbreviation in abbreviations_populations:
                if abbreviations_populations[abbreviation] > max_population:
                    corrected_abbreviation = abbreviation
                    max_population = abbreviations_populations[abbreviation]

            print(f"Spelling correction problem: multiple abbreviation candidates for {predicted_abbreviation} have the same score: {keys_candidates}. {corrected_abbreviation} is chosen")
            return corrected_abbreviation

    def set_costs(self):
        self.insert_costs = np.ones(128, dtype=np.float64) 
        self.delete_costs = np.ones(128, dtype=np.float64)
        self.substitute_costs = np.ones((128, 128), dtype=np.float64)

        # 23.03.24
        self.substitute_costs[ord('{'), ord('(')] = 0.4
        self.substitute_costs[ord('}'), ord(')')] = 0.4
        self.substitute_costs[ord('j'), ord('i')] = 0.4
        self.substitute_costs[ord('8'), ord('B')] = 0.4
        self.substitute_costs[ord('I'), ord('l')] = 0.4
        self.substitute_costs[ord('1'), ord('l')] = 0.4
        self.substitute_costs[ord('O'), ord('2')] = 0.7

        self.delete_costs[ord('.')] = 0.5
        self.delete_costs[ord(',')] = 0.5
        self.delete_costs[ord(';')] = 0.5
        self.delete_costs[ord(':')] = 0.5
        self.delete_costs[ord('"')] = 0.5
        
        # 22.03.24
        self.insert_costs[ord('2')] = 0.6
        self.insert_costs[ord('3')] = 0.8
        self.insert_costs[ord('C')] = 0.8
        self.insert_costs[ord('H')] = 0.8        
        self.insert_costs[ord('(')] = 0.8
        self.insert_costs[ord(')')] = 0.8

        self.delete_costs[ord(' ')] = 0.5

        self.substitute_costs[ord('0'), ord('O')] = 0.1
        self.substitute_costs[ord('o'), ord('O')] = 0.4
        self.substitute_costs[ord('D'), ord('O')] = 0.5
        self.substitute_costs[ord('Q'), ord('O')] = 0.5
        self.substitute_costs[ord('O'), ord('C')] = 0.7
        self.substitute_costs[ord('c'), ord('C')] = 0.4
        self.substitute_costs[ord('D'), ord('C')] = 0.5
        self.substitute_costs[ord('d'), ord('c')] = 0.5
        self.substitute_costs[ord('C'), ord('c')] = 0.8
        self.substitute_costs[ord('.'), ord('c')] = 0.4
        self.substitute_costs[ord('o'), ord('c')] = 0.5
        self.substitute_costs[ord('n'), ord('N')] = 0.4
        self.substitute_costs[ord('h'), ord('H')] = 0.4
        self.substitute_costs[ord('K'), ord('H')] = 0.5
        #self.substitute_costs[ord(' '), ord('H')] = 0.5
        self.substitute_costs[ord('s'), ord('S')] = 0.4
        self.substitute_costs[ord(' '), ord('2')] = 0.6
        self.substitute_costs[ord(','), ord('2')] = 0.4
        self.substitute_costs[ord('z'), ord('2')] = 0.5
        self.substitute_costs[ord('-'), ord('2')] = 0.4
        self.substitute_costs[ord('g'), ord('8')] = 0.5
        self.substitute_costs[ord('g'), ord('3')] = 0.6
        self.substitute_costs[ord('z'), ord('5')] = 0.6
        self.substitute_costs[ord('3'), ord('5')] = 0.8
        self.substitute_costs[ord('3'), ord('5')] = 0.8

        # Added from OSRA's list:
        self.insert_costs[ord('n')] = 0.8

        self.substitute_costs[ord('i'), ord('l')] = 0.5
        self.substitute_costs[ord('f'), ord('l')] = 0.5
        self.substitute_costs[ord('M'), ord('N')] = 0.5
        self.substitute_costs[ord('H'), ord('N')] = 0.8
        self.substitute_costs[ord('l'), ord('M')] = 0.5
        self.substitute_costs[ord('U'), ord('u')] = 0.5
        self.substitute_costs[ord('l'), ord('t')] = 0.5
        self.substitute_costs[ord('8'), ord('e')] = 0.5
        self.substitute_costs[ord('B'), ord('e')] = 0.5
        self.substitute_costs[ord('R'), ord('e')] = 0.5
        self.substitute_costs[ord('5'), ord('S')] = 0.5
        self.substitute_costs[ord('8'), ord('S')] = 0.5
        self.substitute_costs[ord('Z'), ord('z')] = 0.5
        self.substitute_costs[ord('l'), ord('i')] = 0.5
