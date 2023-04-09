from transformers import AutoProcessor, TFAutoModelForQuestionAnswering
import tensorflow as tf
import math
import os
import json
import torch
import pickle
from PIL import Image

def build_up_KG(J_FILE):
    
    with open(J_FILE) as j:
        data = json.load(j)
    
    for d in range(0, len(data['data'])):
        extract_info(data['data'][d]['questionId'], data['data'][d]['question'], data['data'][d]['page_ids'])

def extract_info(QID, question, pids):
    
    for p in pids:
        if p in pages:
            # extract each OCR bboxes, words and image in eact line
            info = preprocess_dataset + p + '/info.json'
            ocr = '/home/grads/tingchih/dataset/DocVQA_task4/ocr/' + p + '.json'
            
            if os.path.exists(info) == True and os.path.exists(ocr) == True:
                
                Q_folder = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/text_image_KG/' + str(QID) + '/'
                if os.path.exists(Q_folder) == False:
                    os.mkdir(Q_folder)
                
                p_folder = Q_folder + p + '/'
                if os.path.exists(p_folder) == False:
                    os.mkdir(p_folder)
                
                with open(info) as i:
                    info_data = json.load(i)
            
                with open(ocr) as o:
                    ocr_data = json.load(o)
            
                for k, v in info_data.items():
                    if int(k) < len(ocr_data['LINE']):
                        words = v['Text']
                        image_filename = '/home/grads/tingchih/dataset/DocVQA_task4/' + v['Image']
                        image = Image.open(image_filename)
                        original_BBOX = ocr_data['LINE'][int(k)]['Geometry']['BoundingBox'] #dict
                        normalize = normalize_BBOX(original_BBOX)
                        
                        tensor_node = obtain_encoding(image, question, [words], [normalize])
                        torch_filename = p_folder + str(k) + '.pt'
                        torch.save(tensor_node, torch_filename)
    
    print('[FINISH] ', QID, pids)

def normalize_BBOX(original):
    # {"BoundingBox": {"Width": , "Height": , "Left": , "Top": }
    
    xmin, ymin = original["Left"], original["Top"]
    xmax, ymax = original["Left"] + original["Width"], original["Top"] + original["Height"]
    
    xmin_norm = math.floor(xmin / original["Width"])
    ymin_norm = math.floor(ymin / original["Height"])
    xmax_norm = math.floor(xmax / original["Width"])
    ymax_norm = math.floor(ymax / original["Height"])
    #print(original)
    #print(xmin_norm, ymax_norm, xmax_norm, ymin_norm)
    ans = [xmin_norm, ymax_norm, xmax_norm, ymin_norm]
    return ans

def obtain_encoding(image, question, words, boxes):

    encoding = processor(image, question, words, boxes=boxes, return_tensors="tf")
    start_positions = tf.convert_to_tensor([1])
    end_positions = tf.convert_to_tensor([3])
    
    encoding = processor(image, question, words, boxes=boxes, return_tensors="tf")
    start_positions = tf.convert_to_tensor([1])
    end_positions = tf.convert_to_tensor([3])
    outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)
    
    return outputs

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = TFAutoModelForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")
train_root = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/train.json'
val_root = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/val.json'
test_root = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/test.json'
preprocess_dataset = '/home/grads/tingchih/dataset/DocVQA_task4/preprocess_dataset/'
pages = os.listdir(preprocess_dataset)

build_up_KG(val_root)