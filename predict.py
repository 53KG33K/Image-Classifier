import torch
import argparse
from utils import process_image, load_checkpoint, class_to_label

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to the JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def predict(image_path, model, topk, device):
    pre_processed_image = torch.from_numpy(process_image(image_path))
    pre_processed_image = torch.unsqueeze(pre_processed_image, 0).to(device).float()
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        log_ps = model.forward(pre_processed_image)
    
    ps = torch.exp(log_ps)
    top_ps, top_idx = ps.topk(topk, dim=1)
    list_ps = top_ps.tolist()[0]
    list_idx = top_idx.tolist()[0]
    classes = []
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    for idx in list_idx:
        classes.append(idx_to_class[idx])
    
    return list_ps, classes

def main():
    args = get_input_args()
    image_path = args.input
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(checkpoint)
    probabilities, classes = predict(image_path, model, top_k, device)
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f
