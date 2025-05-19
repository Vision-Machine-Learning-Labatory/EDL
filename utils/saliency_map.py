import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_saliency(model, image, label, save_path, edl=False):
    """
    EDL 또는 Softmax 모델에 대해 Saliency Map 생성 및 저장.
    EDL의 경우, 4가지 output type 모두 시각화.

    Args:
        model: 학습된 모델
        image: 입력 이미지 (1, C, H, W)
        label: 실제 클래스
        save_path: 저장할 base path (e.g., .../cls3.png)
        edl: 모델이 EDL인지 여부
    """
    image = image.clone().detach().to(image.device)
    image.requires_grad_()
    model.zero_grad()

    orig = image.detach().squeeze().cpu().numpy()

    if edl:
        output = model(image)
        saliency_maps = [orig]  # 첫 번째는 원본
        titles = ["Original"]
        for target_type in ["evidence", "belief", "uncertainty", "probability"]:
            model.zero_grad()
            image.grad = None

            if output[target_type].size(1) == 1:
                # EDL: uncertainty
                score = output[target_type][0]
            else:
                # EDL: evidence, belief, probability
                score = output[target_type][0, label]
            
            # print(f"{target_type} score: {score.item()}")
            score.backward(retain_graph=True)

            saliency = image.grad.data.abs()
            saliency, _ = torch.max(saliency, dim=1)  # (1, H, W)
            saliency = saliency.squeeze().cpu().numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

            saliency_maps.append(saliency)
            titles.append(f"{target_type.capitalize()}")

        # 저장
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            axs[i].imshow(orig, cmap='gray')
            if i > 0:
                axs[i].imshow(saliency_maps[i], cmap='hot', alpha=0.5)
            axs[i].axis('off')
            axs[i].set_title(titles[i],fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    else:
        # Softmax: 단일 score 기준
        output = model(image)
        score = output[0, label]
        score.backward()

        saliency = image.grad.data.abs()
        saliency, _ = torch.max(saliency, dim=1)  # (1, H, W)
        saliency = saliency.squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        orig = image.detach().squeeze().cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(orig, cmap='gray')
        axs[0].axis('off')
        axs[0].set_title("Original")

        axs[1].imshow(orig, cmap='gray')
        axs[1].imshow(saliency, cmap='hot', alpha=0.5)
        axs[1].axis('off')
        axs[1].set_title(f"Saliency (class {label})")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
