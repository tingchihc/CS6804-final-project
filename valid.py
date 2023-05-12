import sys
import torch
import logging
from config import root, device, checkpoint_dir, load_latest
from model import CustomModel
from competition_text_only_KG_val_dgl import build_up_nx_KG

#now we will Create and configure logger 
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

# Load latest checkpoint
_, custom_model = load_latest()
if not custom_model:
    print(f"No checkpoints in directory: {checkpoint_dir}", file=sys.stderr)
    sys.exit(1)

# Iterate validation dataset
count = 0
t, f = 0, 0
with torch.no_grad():
    for target_text, target_page, graph in build_up_nx_KG(root % 'val'):
        text, page = custom_model(graph.to(device))
        #logger.info("Pred:", page, text)
        #logger.info("True:", target_page, target_text[0])
        #print("Pred:", page, text)
        #print("True:", target_page, target_text[0])
        if target_page == page:
            t += 1
        else:
            f += 1
        count += 1
        if not count % 100:
            print(f"{count}:", t/(t+f))
            logger.info(str(t/(t+f)))
print("Final:", t/(t+f))
logger.info(str(t/(t+f)))