import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_c)
        self.conv_1 = nn.Conv2d(in_c, out_c, kernel_size= 3, padding= 1)

        self.groupnorm_2 = nn.GroupNorm(32, out_c)
        self.conv_1 = nn.Conv2d(out_c, out_c, kernel_size= 3, padding= 1)

        if in_c == out_c:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_c, out_c, kernel_size= 1,padding= 0)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)

        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        x = x + self.residual_layer(residue)
        return x
    
class VAE_AttentionBLock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self , x : torch.Tensor) -> torch.Tensor:

        residue = x

        n , c, h, w = x.shape

        x = x.view(n,c,h*w)

        x = x.transpose(-1,-2) # (bs,h*w,c)

        x= self.attention(x)

        x = x.transpose(-1,-2)

        x = x.view(n,c,h,w)

        x += residue

        return x
    
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size= 1, padding= 0),

            nn.Conv2d(4,4,kernel_size= 1, padding= 1),

            VAE_ResidualBlock(512,512),
            VAE_AttentionBLock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512), #(bs,512,h/8,w/8)

            nn.Upsample(scale_factor=2), #(bs,512,h/4,w/4)

            nn.Conv2d(512,512,kernel_size= 3, padding = 1),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            nn.Upsample(scale_factor=2), #(bs,512,h/2,w/2)

            

            nn.Conv2d(512,512,kernel_size= 3, padding = 1),
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),

            nn.Upsample(scale_factor=2), #(bs,512,h,w)

            nn.Conv2d(256,256,kernel_size= 3, padding = 1),
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32,128),
            nn.SiLU(),
            nn.Conv2d(128,3,kernel_size= 3, padding= 1)

        )

    def forward(self , x : torch.Tensor) -> torch.Tensor:
        # x: (bs, 4,h/8,w/8)

        x /= 0.18215

        for module in self:
            x = module(x)
        
        return x