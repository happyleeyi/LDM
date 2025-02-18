import os
import numpy as np
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import datasets,transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns
from Lpips import VGG16LPIPS



class trainer_VQ():
    def __init__(self, device, n_epochs, lr, model):
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = model.to(device)
        self.perceptual_loss_fn = VGG16LPIPS().to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr,amsgrad=False)
        
    def train(self, train_dataloader, model_name):
        for epoch in range(self.n_epochs):
            train_loss = 0
            per_loss_sum=0
            
            for i, batch in enumerate(tqdm(train_dataloader)):
                # forward
                x = batch['pixel_values']

                x = x.to(self.device)
                reconstructed, _, loss = self.model(x)

                #perceptual loss
                per_loss = self.perceptual_loss_fn(x, reconstructed)

                loss += per_loss

                # backprop and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                per_loss_sum += per_loss.item()
                train_loss += loss.item()

            print(f'===> Epoch: {epoch+1} Average Train Loss: {train_loss/len(train_dataloader.dataset)}, per_loss: {per_loss_sum/len(train_dataloader.dataset)} ')
        torch.save(self.model.state_dict(), model_name)

    def test(self, x):
        x = x.to(self.device)
        x_reconst, _, _ = self.model(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x_reconst = x_reconst.permute(0, 2, 3, 1).contiguous()


        fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2행 num_images열

        # 원본 이미지 출력
        for i in range(5):
            axes[0, i].imshow(x.cpu()[i]*0.5+0.5)
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')

        # 재구성된 이미지 출력
        for i in range(5):
            axes[1, i].imshow(x_reconst.cpu().detach().numpy()[i]*0.5+0.5)
            axes[1, i].set_title(f"Reconstructed {i+1}")
            axes[1, i].axis('off')

        # 레이아웃 조정 및 출력
        plt.tight_layout()
        plt.show()


