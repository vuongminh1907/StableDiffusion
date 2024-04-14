import torch
from torch import nn
from torch.nn import functional as F
from decodeder import VAE_AttentionBLock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #(bs,c,h,w) -> (bs,128,h,w)
            nn.Conv2d(3,128, kernel_size= 3, padding= 1),
            #(bs,128,h,w) -> (bs,128,h,w)
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            #(bs,128,h,w) -> (bs,128,h/2,w/2)
            nn.Conv2d(128,128, kernel_size= 3,stride=2, padding= 0),
            VAE_ResidualBlock(128,256),
            #(bs,256,h/2,w/2)
            VAE_ResidualBlock(256,256),
            nn.Conv2d(256,256, kernel_size= 3,stride=2, padding= 0),    #(bs,256,h/4,w/4)
            VAE_ResidualBlock(256,512),
            VAE_ResidualBlock(512,512),

            nn.Conv2d(256,256, kernel_size= 3,stride=2, padding= 0),    #(bs,256,h/8,w/8)

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            VAE_AttentionBLock(512),

            VAE_ResidualBlock(512,512),
            
            nn.GroupNorm(32,512), #(bs,512,h/8,w/8)  num_group = 32
            nn.SiLU(),
            
            nn.Conv2d(512,8, kernel_size= 3, padding= 1),
            nn.Conv2d(8,8, kernel_size= 3, padding= 0),    #(bs,8,h/8,w/8)
        )
    def forward(self, x: torch.Tensor, noise : torch.Tensor) -> torch.Tensor:
        # x: (bs, c, h, w)
        # noise: (bs, 8, h/8, w/8)
        
        for module in self:
            if getattr(module, 'stride', None ) == (2,2):
                x=F.pad(x,(0,1,0,1))
            x = module(x)
        mean, log_variance = torch.chunk(x,2,dim =1) # 2 * (bs,4,h/8,w/8)
        
        log_variance = torch.clamp(log_variance, -30, 20) # kep log_varience in range -30 -> 20
        variance = log_variance.exp()

        stdev = variance.sqrt()

        # Z =N01 -> N(mean,variance)=X
        x = mean + stdev * noise

        x*= 0.18215

        return x
        
        