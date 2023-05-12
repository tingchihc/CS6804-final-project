import torch, os

# Hyperparameters
MAX_QUE = 64
DIM_HID = 16
MAX_ANS = 8
ETA = 0.1

# Data paths
root = '/home/grads/tingchih/dataset/DocVQA_task4/competition_dataset/%s.json'
preprocess_dataset = '/home/grads/tingchih/dataset/DocVQA_task4/preprocess_dataset/'

# GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")          # a CUDA device object
    print("Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Checkpoints
checkpoint_dir = "checkpoints2"
ckpt_freq = 1000
def load_latest():
    try:
        ckpt_num = max(map(int, os.listdir(checkpoint_dir)))
        checkpoint = torch.load(f"{checkpoint_dir}/{ckpt_num}", map_location=device)
        return ckpt_num, checkpoint['model']
    except ValueError:
        return -1, None

