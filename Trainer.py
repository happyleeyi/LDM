import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm.auto import tqdm

from Basic import num_to_groups

from Diffusion import Diffusion
from util import save_checkpoint, load_checkpoint, EMA


class Trainer():
    def __init__(self, device, dataloader, image_size, channels, batch_size, timesteps, lr, ckpt, vae, ldm):
        self.device = device
        self.dataloader = dataloader
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.model = ldm.to(self.device)
        self.lr = lr
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.ema = EMA(self.model, decay=0.99)
        if ckpt is not None:
            load_checkpoint(self.model, self.ema, self.optimizer, ckpt)
        self.diffusion = Diffusion(self.timesteps,0.0015,0.0195)
        self.vae = vae

    def training(self, load_ckpt, save_ckpt, epochs):
        self.model.to(self.device)

        if load_ckpt is not None:
            load_checkpoint(self.model, self.ema, self.optimizer, load_ckpt)

        scaler = torch.amp.GradScaler('cuda')

        self.model.train()

        for epoch in range(epochs):
            loss_sum=0
            for step, batch in enumerate(tqdm(self.dataloader)):
                self.optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(self.device)
                batch = self.vae.encode(batch)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                with torch.amp.autocast('cuda'):
                    loss = self.diffusion.p_losses(self.model, batch, t, loss_type="huber")

                loss_sum+=loss.item()

                if step % 100 == 0:
                    print("Epoch, Loss:", epoch, loss_sum/100)
                    loss_sum=0

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                #loss.backward()
                #optimizer.step()
                self.ema.update()
        save_checkpoint(self.model, self.ema, self.optimizer, save_ckpt)
    def sampling(self, check, t, dt):
        self.model.to(self.device)
        self.ema.apply_shadow()
        self.model.eval()
        with torch.no_grad():
            samples = self.diffusion.sample(check, t, dt, self.model, image_size=self.image_size, batch_size=1, channels=self.channels)
        self.ema.restore()
        return samples
    def save_sample_time(self, check, t, dt, save_name=None):
        samples = self.sampling(check, t, dt)
        samples_img, _ = self.vae.quantize(torch.tensor(samples[-1]).to(self.device))
        samples_img = self.vae.decode(samples_img).cpu().detach().numpy()[0]/2+0.5
        
        np.save('samples.npy', np.array(samples))
        if save_name is not None:
            plt.imshow(np.fliplr(np.rot90(np.transpose(samples_img), 3)))
            plt.savefig(save_name)
        #plt.show()
