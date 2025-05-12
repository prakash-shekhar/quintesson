import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import math

class SimpleDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        
        # Create noise schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-calculate values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to x_0 to get x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Move to same device as x_start
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device)
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        """
        Reverse process: p(x_{t-1} | x_t)
        """
        # Predict noise
        noise_pred = model(x_t, t)
        
        # Compute x_0 from x_t and predicted noise
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].to(x_t.device)[:, None, None, None]
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].to(x_t.device)[:, None, None, None]
        
        pred_original_sample = sqrt_recip_alphas_t * (x_t - sqrt_recipm1_alphas_cumprod_t * noise_pred)
        
        if clip_denoised:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute mean and variance
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].to(x_t.device)[:, None, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t].to(x_t.device)[:, None, None, None]
        beta_t = self.betas[t].to(x_t.device)[:, None, None, None]
        
        pred_prev_sample = (
            sqrt_recip_alphas_t * alpha_cumprod_prev_t * x_t +
            sqrt_recip_alphas_t * beta_t * pred_original_sample
        ) / alpha_cumprod_t
        
        variance = self.posterior_variance[t].to(x_t.device)
        if t[0] == 0:
            variance = 0
            
        return pred_prev_sample, variance
    
    def p_sample(self, model, x_t, t):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t)
        """
        mean, variance = self.p_mean_variance(model, x_t, t)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = ((t != 0).float()).view(-1, *([1] * (len(x_t.shape) - 1)))
        
        sample = mean + nonzero_mask * torch.sqrt(variance[:, None, None, None]) * noise
        return sample
    
    def p_sample_loop(self, model, shape, device):
        """
        Full reverse process: sample from noise to image
        """
        device = next(model.parameters()).device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling'):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
            imgs.append(img.cpu())
            
        return imgs

# Simple U-Net for denoising
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.features = features
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Encoder (down)
        self.encoder = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        
        channels = in_channels
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(channels, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            self.down_convs.append(nn.Conv2d(feature, feature, 2, stride=2))
            channels = feature
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, 3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, 3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (up)
        self.decoder = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        features = [features[-1]*2] + features[::-1]
        for i in range(len(features)-1):
            self.up_convs.append(
                nn.ConvTranspose2d(features[i], features[i+1], 2, stride=2)
            )
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(features[i+1]*2, features[i+1], 3, padding=1),
                    nn.BatchNorm2d(features[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features[i+1], features[i+1], 3, padding=1),
                    nn.BatchNorm2d(features[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final layer
        self.final_conv = nn.Conv2d(features[-1], out_channels, 1)
        
    def forward(self, x, t):
        # Embed timestep
        t_emb = self.time_mlp(t.float().unsqueeze(1))
        t_emb = t_emb.view(t_emb.size(0), t_emb.size(1), 1, 1)
        
        # Encoder
        skip_connections = []
        for encoder, down in zip(self.encoder, self.down_convs):
            x = encoder(x)
            skip_connections.append(x)
            x = down(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = x + t_emb  # Add timestep embedding
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for decoder, up, skip in zip(self.decoder, self.up_convs, skip_connections):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.final_conv(x)

# Training function
def train_diffusion(model, diffusion, dataloader, epochs=100, lr=1e-4, device='cuda'):
    """Train the diffusion model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
            data = data.to(device)
            batch_size = data.size(0)
            
            # Sample timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # Sample noise
            noise = torch.randn_like(data)
            
            # Add noise to data
            x_noisy = diffusion.q_sample(data, t, noise)
            
            # Predict noise
            noise_pred = model(x_noisy, t)
            
            # Compute loss
            loss = criterion(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model

# Example usage
if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # Initialize models
    unet = SimpleUNet(in_channels=3, out_channels=3)
    diffusion = SimpleDiffusion(timesteps=1000)
    
    # Train the model
    model = train_diffusion(unet, diffusion, dataloader, epochs=20, device=device)
    
    # Sample some images
    model.eval()
    with torch.no_grad():
        samples = diffusion.p_sample_loop(model, (16, 3, 32, 32), device)
    
    # Visualize the denoising process
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    timesteps_to_show = [999, 900, 700, 500, 300, 100, 50, 10, 5, 1, 0]
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            if idx < len(timesteps_to_show):
                t = timesteps_to_show[idx]
                if t < len(samples):
                    img = samples[999-t][0]  # Show first image from batch
                    img = img.cpu().numpy().transpose(1, 2, 0)
                    img = (img + 1) / 2  # Denormalize to [0, 1]
                    img = np.clip(img, 0, 1)
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f't={t}')
                    axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('diffusion_process.png')
    plt.show()