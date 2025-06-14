import yaml
import os
import torch
import torch.nn as nn
from datetime import datetime
import random
import json

from models import get_model
from data_loader import get_data_loader
from utils.loss import get_loss_function, kl_divergence
from utils.saliency_map import visualize_saliency
# from utils.grad_cam import visualize_gradcam
from evaluate import evaluate


# config 로딩
with open("configs/visualize.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로딩 (batch_size=1)
loader = get_data_loader(config, train=False)
num_classes = config["model_config"]["model"]["num_classes"]

# ✅ 랜덤 시, 클래스별 타겟 인덱스를 미리 선택
target_indices = [[] for _ in range(num_classes)]
if config["visualization"]["random"]["is_random"]:
    seed = config["visualization"]["random"].get("seed", None)
    if seed is not None:
        random.seed(seed)
    
    class_indices = [[] for _ in range(num_classes)]

    for idx, (x, y) in enumerate(loader):
        label = y.item()
        class_indices[label].append(idx)

    for c in range(num_classes):
        target_idx = random.choice(class_indices[c])
        target_indices[c].append(target_idx)
else:
    target_indices = config["visualization"]["random"]["target_indices"]

    if type(target_indices) == int:
        target_indices = [target_indices for _ in range(num_classes)]

# 모델 생성 및 weight 로드
edl = config["model_config"]["edl"]["enabled"]
model = get_model(config["model_config"]).to(device)
model.load_state_dict(torch.load(config["model_config"]["checkpoint_path"], map_location=device))
model.eval()

# ✅ 시각화 저장 경로
target_type = "edl" if edl else "softmax"
timestamp = datetime.now().strftime("%y%m%d_%H%M")
method = config["visualization"]["method"]
base_dir = config["visualization"]["save_path"]
output_dir = os.path.join(base_dir, f"{method}_{target_type}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# ✅ 시각화 실행 (stream 방식)
seen = set()
class_counts = [0 for _ in range(num_classes)]
outputs = [None for _ in range(num_classes)]

for global_idx, (x, y) in enumerate(loader):
    label = y.item()
    current_idx = class_counts[label]

    if current_idx == target_indices[label] and label not in seen:
        x = x.to(device)
        y = y.to(device)
        
        if edl:
            output = model(x)["evidence"].detach().cpu().numpy()
            outputs[label] = output
        else:
            output = model(x).detach().cpu().numpy()
            outputs[label] = output
        save_path = os.path.join(output_dir, f"cls{label}.png")

        if method == "saliency":
            visualize_saliency(model, x, y, save_path, edl)
        # else:
        #     visualize_gradcam(model, x, y, config["visualization"], save_path, edl)

        print(f"✅ Class {label} 시각화 완료 → {save_path}")
        seen.add(label)

        if len(seen) == num_classes:
            break

    class_counts[label] += 1

json_path = os.path.join(output_dir, "outputs.json")

with open(json_path, 'a') as f:
    for i in range(num_classes):
        f.write('{"class":' + str(i) + ',\n')
        f.write('"evidence": ' + json.dumps(outputs[i].tolist(), separators=(',', ': ')) + '}\n')