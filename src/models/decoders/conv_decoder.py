import torch.nn as nn, torch, numpy as np

class Decoder(nn.Module):
    def __init__(self, layer_params: dict):
        super().__init__()
        self.input_height = layer_params["input_height"]
        self.input_width = layer_params["input_width"]
        intermediate_dims = layer_params["intermediate_dims"]  # [256, 128]
        latent_dim = layer_params["latent_dim"]
        
        dims = intermediate_dims  + [1]  # e.g. [128, 256, 1]
        spatial_dims = self._compute_spatial_dims(n_layers=len(intermediate_dims))
        flattened_size = intermediate_dims[0] * spatial_dims[0][0] * spatial_dims[0][1]
        self.fc = nn.Linear(latent_dim, flattened_size)
        
        layers = [nn.Unflatten(1, (intermediate_dims[0], spatial_dims[0][0], spatial_dims[0][1]))]
        
        for i, (in_ch, out_ch) in enumerate(zip(dims, dims[1:])):
            target_h, target_w = spatial_dims[i + 1]
            current_h, current_w = spatial_dims[i]
            
            # output_padding makes up for floor division during encoding
            out_pad_h = target_h - (current_h * 2 - 1) - 1 + 1  # simplified: target - (stride*input - 1)
            out_pad_w = target_w - (current_w * 2 - 1) - 1 + 1
            
            # Clamp to valid range [0, stride-1] = [0, 1]
            out_pad_h = max(0, min(1, out_pad_h))
            out_pad_w = max(0, min(1, out_pad_w))

            layers.append(nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size=3, stride=2, padding=1,
                output_padding=(out_pad_h, out_pad_w)
            )) #type: ignore

            if out_ch != 1:  # no BatchNorm or activation on final output layer
                layers.append(nn.BatchNorm2d(out_ch)) #type: ignore
                layers.append(nn.LeakyReLU(inplace=True)) #type: ignore
        
        layers.append(nn.Sigmoid()) # type: ignore
        self.net = nn.Sequential(*layers)
        
    def _compute_spatial_dims(self, n_layers):
        """
        Simulate the encoder's spatial downsampling to get H and W at each stage.
        Returns list of (H, W) from deepest to shallowest (decoder order).
        e.g. for input 64x91 with 3 layers: [(8,12), (16,23), (32,46), (64,91)]
        """
        dims = [(self.input_height, self.input_width)]
        h, w = self.input_height, self.input_width
        for _ in range(n_layers):
            h = (h + 1) // 2  # equivalent to ceil(h / stride) with stride=2, padding=1
            w = (w + 1) // 2
            dims.append((h, w))

        dims.reverse()  # decoder order: smallest first
        
        return dims
        
    def forward(self, z):
        h = self.fc(z)
        
        return self.net(h)
    
if __name__ == "__main__":
    enc = Decoder({"input_height": 64, "input_width": 91, "intermediate_dims": [128, 64, 32], "latent_dim": 32})
    
    print(enc)