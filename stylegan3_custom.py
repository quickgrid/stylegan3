import torch
import pickle
import matplotlib.pyplot as plt

with open(r'downloads/stylegan3-r-ffhqu-1024x1024.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes

# print(z)
print(z.shape)

c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1], no truncation

np_img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
np_img = np_img[0].cpu().numpy()
plt.imshow(np_img)
plt.show()
