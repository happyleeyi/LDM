# Latent Diffusion Model
논문명 : High-Resolution Image Synthesis with Latent Diffusion Models

## Training
![image](https://github.com/user-attachments/assets/9c1f5429-9611-42c8-96db-9437bf57c196)

- VAE를 통해 이미지를 Latent vector로 압축
- 압축한 vector들에 대해서 DDIM 학습
- VAE는 VQ-VAE를 사용
- VAE loss로 l2 loss + perceptual loss (lpips) 사용

## Sampling
- 학습한 DDIM으로 latent vector 생성
- 생성한 latent vector를 VAE로 decoding
- VAE 성능이 생각보다 최종 성능에 영향을 주는 듯함

## Result - CelebHQ(256*256) 학습
- 통상적으로 8배 압축이 가장 성능이 잘나옴
- 하지만 U-net 크기의 한계인지 제대로 학습이 잘 안되어서 포기
- cherry picking한 결과는 다음과 같음
- ![image](https://github.com/user-attachments/assets/e58cc30b-7330-4eb9-bcc9-d7d0db9e037f)
- ![image](https://github.com/user-attachments/assets/ae4f9d6c-09c2-41ff-9d49-9db52cb99c9e)

