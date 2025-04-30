import yaml
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from models import get_model
from data_loader import get_data_loader
from utils.loss import get_loss_function, kl_divergence
from evaluate import evaluate


if __name__ == "__main__":
    with open('configs/train.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로딩
    train_loader = get_data_loader(config, train=True)
    valid_loader = get_data_loader(config, train=False)
    # 모델 생성
    edl = config['edl']['enabled']
    criterion = get_loss_function(config['edl']['loss_type']) if edl else nn.CrossEntropyLoss()
    model = get_model(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    train_type = "edl" if edl else "softmax"
    dataset_name = config['dataset']['name']
    now = datetime.now()
    date_time = now.strftime("%y%m%d_%H:%M")
    experiment_name = f"{train_type}_{dataset_name}_{date_time}"

    log_dir = os.path.join("runs", experiment_name)
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_root = config['train']['checkpoint_path']
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_root,experiment_name), exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_root,experiment_name,"model.pth")
    config_save_path = os.path.join(checkpoint_root,experiment_name,"train.yaml")

    epochs = config['train']['epochs']

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            if edl:
                loss = criterion(output['alpha'], y, config['model']['num_classes'], kl_scale=min(1, epoch / epochs))
                pred = torch.argmax(output['probability'], dim=1)
            else:
                loss = criterion(output, y)
                pred = torch.argmax(output, dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (pred == y).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        val_acc, val_loss = evaluate(model, valid_loader, config, criterion, edl, device)

        # TensorBoard 기록
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # 모델과 config 저장
    torch.save(model.state_dict(), checkpoint_path)

    # config 파일 저장 (현재 config dict를 YAML로 다시 저장)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    writer.close()