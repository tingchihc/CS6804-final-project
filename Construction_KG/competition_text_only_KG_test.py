import os
import json
import networkx as nx
import pickle
import matplotlib.pyplot as plt

def build_up_nx_KG(J_FILE, split_set):
    
    with open(J_FILE) as jj:
        data = json.load(jj)

    for c in range(0, len(data['data'])):
        #print(data['data'][c]['page_ids'], data['data'][c]['questionId'])
        extract_info_from_preprocess_dataset(data['data'][c]['page_ids'], data['data'][c]['questionId'], split_set)


def extract_info_from_preprocess_dataset(pids, QID, split_set):
    
    node = {} #{page_id: [n1,n2,n3,...], ...}
    
    for p in pids:
        page_number = p.split('_')[1]
        node[page_number] = []
        info = preprocess_dataset + p + '/info.json'
        ocr = '/home/grads/tingchih/dataset/DocVQA_task4/ocr/' + p + '.json'

        if os.path.exists(info) == True:

            with open(ocr) as jj:
                ocr_data = json.load(jj)
            
            with open(info) as jj:
                info_data = json.load(jj)
        
            for k, v in info_data.items():
                if int(k) < len(ocr_data['LINE']):
                    node[page_number].append(v['Text'])
    
        #print('page_number: ', page_number)

    generate_text_only_KG(node, QID, split_set)

def generate_text_only_KG(node_dict, qid, split_set):
    
    G = nx.Graph()
    G.clear()
    page_keys = list(node_dict.keys())
    pk = []
    for p in page_keys:
        pk.append((p, {"color": "red"}))
    G.add_nodes_from(pk, info='page_number')
    
    # page nodes link with page nodes
    for pp in range(0, len(page_keys)):
        if page_keys[pp] != page_keys[-1]:
            G.add_edge(page_keys[pp], page_keys[pp+1])
            
    
    for k, v in node_dict.items():
        G.add_nodes_from(v, color='blue')
    
    print('[NODES] ', list(G.nodes))

    edges = {} # {0:[(n1,n2), (n2,n3)], 1:[]}
    
    for k, v in node_dict.items():
        edges[k] = []
        for vv in range(0, len(v)):
            # nodes link with page node
            edges[k].append((k,v[vv]))
            # nodes link with other nodes
            if v[vv] != v[-1]:
                edges[k].append((v[vv], v[vv+1]))
    
    for ek, ev in edges.items():
        G.add_edges_from(ev)
    

    # save network as pickle file
    filename = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/text_only_KG/' + split_set + '/' + str(qid) + '.p'
    with open(filename, 'wb') as f:
        pickle.dump(G, f)
    
    print('[FINISH] ', split_set, str(qid), G.number_of_nodes(), G.number_of_edges())

    try:
        # visulize_KG
        V_filename = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/text_only_KG/graph_visulization/' + str(qid) + '.png'
        plt.figure(3,figsize=(12,12))
        node_colors = [node[1]['color'] for node in G.nodes(data=True)]
        nx.draw(G, with_labels = True, node_size=90,font_size=8, node_color=node_colors)
        plt.savefig(V_filename)
        plt.clf()
        print('[VISULIZE] ', V_filename)
    except:
        return 0


train_root = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/train.json'
val_root = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/val.json'
test_root = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/test.json'
preprocess_dataset = '/home/grads/tingchih/dataset/DocVQA_task4/preprocess_dataset/'

build_up_nx_KG(test_root, 'test')
