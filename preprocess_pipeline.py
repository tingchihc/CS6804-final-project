from PIL import Image
import json
import math
import os
import cv2
import numpy as np

def extract_single_page_KG(ocr_root):
    ocr_json = os.listdir(ocr_root)
    NO_info = {}
    c_NO_info = 0
    for ocr in ocr_json:
        J_INFO = {}
        ROOT = ocr.split('.')[0]
        file = 'ocr/' + ocr
        image_file = 'images/' + ocr.split('.')[0] + '.jpg'
        with open(file) as f:
            data = json.load(f)
        print(file)
        print(image_file)
        image = cv2.imread(image_file)
        H, W, C = image.shape
        count_J_INFO = 0
        J_file_name = 'info.json'
        
        for k,v in data.items():
            if k == 'LINE':
                for count in range(0, len(v)):
                    bounding_box = {'Width': 0, 'Height': 0, 'Left': 0, 'Top': 0}
                    bounding_box['Width'] = v[count]['Geometry']['BoundingBox']['Width']
                    bounding_box['Height'] = v[count]['Geometry']['BoundingBox']['Height']
                    bounding_box['Left'] = v[count]['Geometry']['BoundingBox']['Left']
                    bounding_box['Top'] = v[count]['Geometry']['BoundingBox']['Top']
                    
                    if os.path.exists('preprocess_dataset/' + ROOT) == False:
                        os.mkdir('preprocess_dataset/' + ROOT)
                    filename = 'preprocess_dataset/' + ROOT + '/' + str(count) + '.jpg'
                    J_INFO[count_J_INFO] = {'Text': v[count]['Text'], 'Image': filename}
                    count_J_INFO += 1
                    extract_img_w_bbox(H, W, bounding_box['Left'], bounding_box['Top'], bounding_box['Width'], bounding_box['Height'], image, filename)
    
            elif k =='WORD':
                for count in range(0, len(v)):
                    bounding_box = {'Width': 0, 'Height': 0, 'Left': 0, 'Top': 0}
                    bounding_box['Width'] = v[count]['Geometry']['BoundingBox']['Width']
                    bounding_box['Height'] = v[count]['Geometry']['BoundingBox']['Height']
                    bounding_box['Left'] = v[count]['Geometry']['BoundingBox']['Left']
                    bounding_box['Top'] = v[count]['Geometry']['BoundingBox']['Top']
                    
                    if os.path.exists('preprocess_dataset/' + ROOT) == False:
                        os.mkdir('preprocess_dataset/' + ROOT)
                    filename = 'preprocess_dataset/' + ROOT + '/' + str(count_J_INFO) + '.jpg'
                    J_INFO[count_J_INFO] = {'Text': v[count]['Text'], 'Image': filename}
                    count_J_INFO += 1
                    extract_img_w_bbox(H, W, bounding_box['Left'], bounding_box['Top'], bounding_box['Width'], bounding_box['Height'], image, filename)
        
        # save json file
        if J_INFO != {}:
            b = json.dumps(J_INFO, indent=4)
            f = open('preprocess_dataset/' + ROOT + '/' + J_file_name, 'w')
            f.write(b)
            f.close()
            print(ocr)
        else:
            NO_info[c_NO_info] = ROOT
            c_NO_info += 1
    
    b = json.dumps(NO_info, indent=4)
    f = open('preprocess_dataset/NO_info.json', 'w')
    f.write(b)
    f.close()

def extract_img_w_bbox(cv2h, cv2w, bbx_L, bbx_T, bbx_W, bbx_H, image, filename):
    left_c = math.floor(bbx_L*cv2w)
    top_c = math.floor(bbx_T*cv2h)
    bb_w = math.floor(bbx_W*cv2w)
    bb_h = math.floor(bbx_H*cv2h)
    
    crop = image[top_c:top_c+bb_h, left_c:left_c+bb_w]
    cv2.imwrite(filename, crop)

TRAIN_JFILE = 'train.json'
VAL_JFILE = 'val.json'
TEST_JFILE = 'test.json'
IMAGE_folder = 'images'
OCR = 'ocr'
ocr_all_files = os.listdir(OCR)
#print('OCR files: ', len(ocr_all_files))

extract_single_page_KG(OCR)