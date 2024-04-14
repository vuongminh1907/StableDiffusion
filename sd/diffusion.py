import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4* n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4* n_embd)
    def forward(self, x):
        x= self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x            #(1,1280)
    

class SwitchSequential(nn.Sequential):
    def forward(self, x, context , time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x,context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x,time)
            else:
                x = layer(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding= 1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merge = nn.GroupNorm(32,out_channels)
        self.conv_merged = nn.Conv2d(out_channels,out_channels,kernel_size=3, padding= 1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1, padding= 0)

    def forward(self, feature, time):
        #feature (bs,in_c,h,w)
        #time (1,1280)
        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # (Batch_Size, Out_Channels, Height, Width)
        # context: (bs, seg_len, dim)

        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        #### Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module([
            # (bs,4,h/8,w/8)
            SwitchSequential(nn.Conv2d(4,320,kernel_size= 3, padding= 1)),
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            
            SwitchSequential(nn.Conv2d(320,320,kernel_size= 3, stride = 2, padding= 1)), # (bs,320,h/16,w/16)
            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(640,640), UNET_AttentionBlock(8,80)),
            
            SwitchSequential(nn.Conv2d(640,640,kernel_size= 3, stride = 2, padding= 1)),    # (bs,640,h/32,w/32)
            SwitchSequential(UNET_ResidualBlock(640,1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8,160)),
            
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size= 3, stride = 2, padding= 1)), # (bs,1280,h/64,w/64)
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
            SwitchSequential(UNET_ResidualBlock(1280,1280))
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_ResidualBlock(1280,1280)
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1920,1280), UNET_AttentionBlock(8,160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920,640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(1280,640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(960,640), UNET_AttentionBlock(8,80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8,40))
        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super.__init__()
        self.groupnorm = nn.GroupNorm(32, in_c)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size= 3, padding= 1)
    def forward(self,x):
        #(bs,320,h/8,w/8)

        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)   # 320 is size
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)

    def forward(self ,latent, context, time):
        # x (bs,4,h/8,w/8)
        # context : (bs, seg_len, dim)
        # time (1, 320)
        time = self.time_embedding(time) 

        # (bs, 320, h/8, w/8)
        output = self.unet(latent,context,time)

        # (bs, 4, h/8, w/8)
        output = self.final(output)

        return output
