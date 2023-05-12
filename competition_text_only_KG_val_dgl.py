import os
import json
import dgl
import torch
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *
from collections import defaultdict
from T5_encoder import T5Encoder

t5_encoder = T5Encoder(max_length=MAX_QUE)


def build_up_nx_KG(J_FILE, skip=0, progress=False):
    
    with open(J_FILE) as jj:
        data = json.load(jj)

    iterable = data['data'][skip:]
    if progress:
        iterable = tqdm(iterable)

    for entry in iterable:
        #print(data['data'][c]['page_ids'], data['data'][c]['questionId'])
        #decoder_test(data['data'][c]['question'],data['data'][c]['answers'])
        yield entry['answers'], entry['answer_page_idx'], extract_info_from_preprocess_dataset(entry['page_ids'], entry['question'])

def extract_info_from_preprocess_dataset(pids, qtext):
    
    node = {} #{page_id: [n1,n2,n3,...], ...}
    
    for p in pids:
        page_number = p.split('_')[1]
        node[page_number] = []
        info = preprocess_dataset + p + '/info.json'
        ocr = '/home/grads/tingchih/dataset/DocVQA_task4/ocr/' + p + '.json'
        
        if os.path.exists(info) == True:

            with open(ocr, 'r') as jj:
                ocr_data = json.load(jj)
            
            with open(info, 'r') as jj:
                info_data = json.load(jj)
        
            for k, v in info_data.items():
                if int(k) < len(ocr_data['LINE']):
                    node[page_number].append(v['Text'])

    return generate_text_only_KG(node, qtext)


def generate_text_only_KG(node_dict, qtext):
    
    d = defaultdict(list)
    
    # add Q-P edges
    for pi, _ in enumerate(node_dict):
        # question node links with page nodes
        d[('P', 'P-Q', 'Q')].append((pi, 0))
        d[('Q', 'P-Q', 'P')].append((0, pi))
        if pi:
            # page nodes link with page nodes
            d[('P', 'P-P', 'P')].append((pi-1, pi))
            d[('P', 'P-P', 'P')].append((pi, pi-1))

    # P-N; N-N
    off = 0
    for pi, ns in enumerate(node_dict.values()):
        for i, _ in enumerate(ns):
            ni = off + i
            # no-des link with page node
            d[('N', 'N-P', 'P')].append((ni, pi))
            d[('P', 'N-P', 'N')].append((pi, ni))
            d[('N', 'N-N', 'N')].append((ni, ni))
            if i:
                # nodes link with other nodes
                d[('N', 'N-N', 'N')].append((ni-1, ni))
                d[('N', 'N-N', 'N')].append((ni, ni-1))
        off += len(ns)
    
    g = dgl.heterograph(d)

    qqq = t5_encoder(qtext).transpose(-1, -2)
    nodes = list(itertools.chain.from_iterable(node_dict.values()))
    g.nodes['Q'].data['feature'] = qqq
    g.nodes['N'].data['feature'] = torch.zeros(len(nodes), *qqq.shape[1:])
    for ni, n in enumerate(nodes):
        g.nodes['N'].data['feature'][ni] = t5_encoder(n).transpose(-1, -2)

    return g

