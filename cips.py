import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size=256, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn((input_dim, mapping_size)) * scale, requires_grad=False)
    
    def forward(self, x):
        x = x.matmul(self.B)
        return torch.sin(x)

class ModulatedFC(nn.Module):
    def __init__(self, in_features, out_features, style_dim):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.modulation = nn.Linear(style_dim, in_features)
        
    def forward(self, x, style):
        style = self.modulation(style).unsqueeze(1)
        x = self.fc(x * style)
        return x

class CIPSGenerator(nn.Module):
    def __init__(self, input_dim, ngf=64, max_resolution=256, style_dim=512, num_layers=8):
        super(CIPSGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.ngf = ngf
        self.max_resolution = max_resolution
        self.style_dim = style_dim
        self.num_layers = num_layers
        
        self.mapping_network = nn.Sequential(
            nn.Linear(input_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        self.fourier_features = FourierFeatures(2, 256)
        self.coord_embeddings = nn.Parameter(torch.randn(1, 512, max_resolution, max_resolution))
        
        self.layers = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        current_dim = 512 + 256  # Fourier features + coordinate embeddings
        
        for i in range(num_layers):
            self.layers.append(ModulatedFC(current_dim, ngf * 8, style_dim))
            if i % 2 == 0 or i == num_layers - 1:
                self.to_rgb.append(ModulatedFC(ngf * 8, 3, style_dim))
            current_dim = ngf * 8
        
    def get_fourier_state(self):
        return self.fourier_features.B.data

    def set_fourier_state(self, state):
        self.fourier_features.B.data = state

    def get_coord_grid(self, batch_size, resolution):
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        x, y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return coords.to(next(self.parameters()).device)
    
    def forward(self, x, target_resolution):
        batch_size = x.size(0)
        
        # Map input to style vector
        w = self.mapping_network(x)
        
        # Generate coordinate grid
        coords = self.get_coord_grid(batch_size, target_resolution)
        coords_flat = coords.view(batch_size, -1, 2)
        
        # Get Fourier features and coordinate embeddings
        fourier_features = self.fourier_features(coords_flat)
        coord_embeddings = F.grid_sample(
            self.coord_embeddings.expand(batch_size, -1, -1, -1),
            coords,
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1).reshape(batch_size, -1, 512)
        
        # Concatenate Fourier features and coordinate embeddings
        features = torch.cat([fourier_features, coord_embeddings], dim=-1)
        
        rgb = 0
        for i, (layer, to_rgb) in enumerate(zip(self.layers, self.to_rgb)):
            features = layer(features, w)
            features = F.leaky_relu(features, 0.2)
            
            if i % 2 == 0 or i == self.num_layers - 1:
                rgb = rgb + to_rgb(features, w)
        
        output = torch.sigmoid(rgb).view(batch_size, target_resolution, target_resolution, 3).permute(0, 3, 1, 2)
        
        # Ensure output is in [-1, 1] range
        output = (output * 2) - 1
        
        return output
    


'''

Context Encoding:
Similar to LazyDiffusion, we can add a context encoder that processes the entire image and mask to produce a compact global context.
Partial Generation:
Modify the generator to only produce content for the masked region, rather than the entire image.
Efficient Conditioning:
Use the compressed context to condition the generation process efficiently.

Context Encoding: We've added a ContextEncoder that processes the entire image and mask to produce a compact global context.
Partial Generation: The generator now only processes the masked region. We use the mask to select only the relevant features for generation.
Efficient Conditioning: We use the compressed context along with the Fourier features and coordinate embeddings to condition the generation process.

Key changes and additions:

A ContextEncoder class that encodes the entire image and mask into a compact representation.
The forward method now takes additional canvas and mask inputs.
We extract context features for the masked region using F.grid_sample.
The generation process now only occurs for masked pixels, reducing computation for partial updates.
The generated content is placed back into the full image at the end.

This "lazy" version of the CIPSGenerator should be more efficient for partial image updates, similar to LazyDiffusion. It will only generate content for the masked region while still maintaining awareness of the global context.
To use this generator effectively, you would need to implement a similar pipeline to LazyDiffusion, where you have a separate encoder that runs once to produce the context, and then the generator can be run multiple times efficiently for different masks or prompts.
'''
class ContextEncoder(nn.Module):
    def __init__(self, input_dim, context_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim + 1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, context_dim, 3, padding=1)
        )
    
    def forward(self, x, mask):
        input = torch.cat([x, mask], dim=1)
        return self.encoder(input)

class LazyCIPSGenerator(nn.Module):
    def __init__(self, input_dim, ngf=64, max_resolution=256, style_dim=512, num_layers=8, context_dim=256):
        super(LazyCIPSGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.ngf = ngf
        self.max_resolution = max_resolution
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        
        self.mapping_network = nn.Sequential(
            nn.Linear(input_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        self.context_encoder = ContextEncoder(3, context_dim)
        
        self.fourier_features = FourierFeatures(2, 256)
        self.coord_embeddings = nn.Parameter(torch.randn(1, 512, max_resolution, max_resolution))
        
        self.layers = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        current_dim = 512 + 256 + context_dim  # Fourier features + coordinate embeddings + context
        
        for i in range(num_layers):
            self.layers.append(ModulatedFC(current_dim, ngf * 8, style_dim))
            if i % 2 == 0 or i == num_layers - 1:
                self.to_rgb.append(ModulatedFC(ngf * 8, 3, style_dim))
            current_dim = ngf * 8
    
    def get_coord_grid(self, batch_size, resolution):
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        x, y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return coords.to(next(self.parameters()).device)
    
    def forward(self, x, canvas, mask, target_resolution):
        batch_size = x.size(0)
        
        # Map input to style vector
        w = self.mapping_network(x)
        
        # Encode context
        context = self.context_encoder(canvas, mask)
        
        # Generate coordinate grid
        coords = self.get_coord_grid(batch_size, target_resolution)
        coords_flat = coords.view(batch_size, -1, 2)
        
        # Get Fourier features and coordinate embeddings
        fourier_features = self.fourier_features(coords_flat)
        coord_embeddings = F.grid_sample(
            self.coord_embeddings.expand(batch_size, -1, -1, -1),
            coords,
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1).reshape(batch_size, -1, 512)
        
        # Get context features for masked region
        context_features = F.grid_sample(
            context,
            coords,
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1).reshape(batch_size, -1, self.context_dim)
        
        # Concatenate Fourier features, coordinate embeddings, and context features
        features = torch.cat([fourier_features, coord_embeddings, context_features], dim=-1)
        
        # Only process masked region
        mask_flat = F.interpolate(mask, size=(target_resolution, target_resolution), mode='nearest').view(batch_size, -1, 1)
        features = features[mask_flat.squeeze(-1) > 0.5]
        
        rgb = 0
        for i, (layer, to_rgb) in enumerate(zip(self.layers, self.to_rgb)):
            features = layer(features, w)
            features = F.leaky_relu(features, 0.2)
            
            if i % 2 == 0 or i == self.num_layers - 1:
                rgb = rgb + to_rgb(features, w)
        
        output = torch.sigmoid(rgb)
        
        # Ensure output is in [-1, 1] range
        output = (output * 2) - 1
        
        # Place generated content back into full image
        full_output = canvas.clone()
        full_output[mask > 0.5] = output
        
        return full_output

# Usage example:
# lazy_cips = LazyCIPSGenerator(input_dim=256)
# latent = torch.randn(1, 256)
# canvas = torch.randn(1, 3, 256, 256)  # Existing image
# mask = torch.zeros(1, 1, 256, 256)
# mask[:, :, 100:150, 100:150] = 1  # Area to be generated
# output = lazy_cips(latent, canvas, mask, target_resolution=256)