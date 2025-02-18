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
from numba import cuda

from Basic import num_to_groups
from Model import Unet
from Trainer import Trainer
from Trainer_VAE import trainer_VQ
from Model_VAE import VQVAE
from Model import Unet

import kagglehub

# Download latest version
#path = kagglehub.dataset_download("crawford/cat-dataset")

#print("Path to dataset files:", path)

# load dataset from the hub
# catface : ./cat_face/cat_face
# celeb : tonyassi/celebrity-1000      #25

dataset = load_dataset(path="tonyassi/celebrity-1000")
trial = 0
model_name = "./celeb_ldm2_1000," + str(trial) + ".pt"
VAE_name = "./celeb_VQVAE2, 50.pt"
image_size = 256
latent_size = int(image_size/4)
embedding_dim = 4
channels = 3
batch_size = 20
timesteps = 1000      #총 time step
lr = 1e-5
lr_vae = 1e-4
epochs_vae = 30
epochs = 20

training_vae_continue=1
testing_vae = 0            #vae testing 할거면 1
training_state_vae = 0     #training 해야되면 1, 모델있으면 0

training_state = 0       #training 단계면 1 sampling 단계면 0
sample_time = 5
save_folder = './'
check = 1                  #sampling 처음이면 1, 아니면 0
dt = 100                    #ddim time step 몇번씩 건너뛸지
gpu = 1                    #gpu 쓸지
repeat = None              #반복해서 sampling 할지
device = "cuda" if torch.cuda.is_available() else "cpu"

######## LDM Model Parameter ###############
down_channels=[224, 224*2, 224*3, 224*4]
mid_channels=[224*4, 224*3]
down_sample=[True, True, True]
attns = [True, True, True]
num_down_layers = 2
num_mid_layers = 2
num_up_layers = 2
z_channels = 4
norm_channels = 32
num_heads = 32
conv_out_channels = 128
time_emb_dim = 512
################################################


############### VAE Model parameter ############
vae_down_channels=[32, 64, 128, 256]
vae_mid_channels=[256, 256]
vae_down_sample=[True, True, False]
vae_attns = [False, False, False]
vae_num_down_layers = 2
vae_num_mid_layers = 2
vae_num_up_layers = 2
vae_z_channels = 4
vae_codebook_size = 8192
vae_norm_channels = 32
vae_num_heads = 4
################################################


transform = Compose([
            #transforms.CenterCrop(256),
            transforms.Resize((image_size,image_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t*2) - 1),
])

# define function 


def transforms(examples):   

   examples["pixel_values"] = [transform(image) for image in examples["image"]]

   del examples["image"]

   return examples

transformed_dataset = dataset.with_transform(transforms)

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
""" dataiter = iter(dataloader)
plt.imshow(np.fliplr(np.rot90(np.transpose(next(dataiter)['pixel_values'][0]/2+0.5), 3)))
plt.show() """

vae = VQVAE(channels, vae_down_channels, vae_mid_channels, vae_down_sample, vae_attns, vae_num_down_layers, vae_num_mid_layers,
             vae_num_up_layers, vae_z_channels, vae_codebook_size, vae_norm_channels, vae_num_heads)

ldm = Unet(z_channels, down_channels, mid_channels, down_sample, attns, num_down_layers, num_mid_layers, num_up_layers, norm_channels, num_heads, conv_out_channels, time_emb_dim)

if training_vae_continue:
   vae.to(device)
   vae_state_dict = torch.load(VAE_name, map_location=device)
   vae.load_state_dict(vae_state_dict)
   #vae.eval()

Trainer_vq = trainer_VQ(device, epochs_vae, lr_vae, vae)

if training_state_vae:
   Trainer_vq.train(dataloader, VAE_name)
   data_iter = iter(dataloader)
   batch = next(data_iter)  # 첫 번째 배치 가져오기

   images = batch["pixel_values"]  # 이미지 데이터 추출


   # 첫 번째 이미지 선택
   first_image = images[0:5]  # 첫 번째 이미지
   Trainer_vq.test(first_image)

if testing_vae:
   data_iter = iter(dataloader)
   batch = next(data_iter)  # 첫 번째 배치 가져오기

   images = batch["pixel_values"]  # 이미지 데이터 추출


   # 첫 번째 이미지 선택
   first_image = images[0:5]  # 첫 번째 이미지
   Trainer_vq.test(first_image)


repeat = int(timesteps/dt)


t = timesteps -1

if training_state:
   if trial == 0:
      model_name = None
   for epoch in range(epochs):
      print(str(trial+epoch+1)+" training start")
      if epoch == 0:
         load_ckpt = model_name
      else :
         load_ckpt = "./celeb_ldm2_1000," + str(trial+epoch) + ".pt"
      save_ckpt = "./celeb_ldm2_1000," + str(trial+epoch+1) + ".pt"
      trainer = Trainer(device, dataloader, latent_size, embedding_dim, batch_size, timesteps, lr, load_ckpt, vae, ldm)
      trainer.training(load_ckpt, save_ckpt, 1)
      t = timesteps -1
      check = 1
      for i in range(repeat):
         print(t)
         if t<dt:
            trainer.save_sample_time(check, t, dt-1, "./fig_"+str(t))
         else:
            trainer.save_sample_time(check, t, dt, "./fig_"+str(t))
         check = 0
         t = t - dt
else:
   trainer = Trainer(device, dataloader, latent_size, embedding_dim, batch_size, timesteps, lr, model_name, vae, ldm)
   for k in range(sample_time):
      t = timesteps -1
      check = 1
      for i in range(repeat):
         print(t)
         if t<dt:
            save_name = save_folder + str(k)
            trainer.save_sample_time(check, t, dt-1, save_name)
         else:
            trainer.save_sample_time(check, t, dt)
         check = 0
         t = t - dt
      