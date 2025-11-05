import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer
    Decomposes weight updates into low-rank matrices: ΔW = BA
    where B is (out_features, r) and A is (r, in_features)
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Ensure LoRA matrices are on the same device as input
        if self.lora_A.device != x.device:
            self.to(x.device)
        # x: (batch, ..., in_features)
        # Compute low-rank update: x @ A^T @ B^T * scaling
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, linear_layer, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)


class Conv2dWithLoRA(nn.Module):
    """Conv2d layer with LoRA adaptation"""
    def __init__(self, conv_layer, rank=4, alpha=1.0):
        super().__init__()
        self.conv = conv_layer
        
        # For conv layers, we apply LoRA to the flattened weight matrix
        out_channels = conv_layer.out_channels
        in_channels = conv_layer.in_channels
        kernel_size = conv_layer.kernel_size[0] * conv_layer.kernel_size[1]
        
        self.lora = LoRALayer(
            in_channels * kernel_size,
            out_channels,
            rank=rank,
            alpha=alpha
        )
        
        # Freeze original weights
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
    
    def forward(self, x):
        # Original conv output
        conv_out = self.conv(x)
        
        # LoRA adaptation
        batch_size, in_c, h, w = x.shape
        
        # Extract patches
        patches = nn.functional.unfold(
            x,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding
        )
        # patches: (batch, in_c * k * k, num_patches)
        
        # Apply LoRA
        patches = patches.transpose(1, 2)  # (batch, num_patches, in_c * k * k)
        lora_out = self.lora(patches)  # (batch, num_patches, out_c)
        lora_out = lora_out.transpose(1, 2)  # (batch, out_c, num_patches)
        
        # Reshape to output size
        out_h = (h + 2 * self.conv.padding[0] - self.conv.kernel_size[0]) // self.conv.stride[0] + 1
        out_w = (w + 2 * self.conv.padding[1] - self.conv.kernel_size[1]) // self.conv.stride[1] + 1
        lora_out = lora_out.view(batch_size, self.conv.out_channels, out_h, out_w)
        
        return conv_out + lora_out


def inject_lora_to_linear(model, rank=4, alpha=1.0, target_modules=None):
    """
    Inject LoRA into all Linear layers in the model
    
    Args:
        model: PyTorch model
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to apply LoRA (None = all)
    """
    if target_modules is None:
        target_modules = ['to_q', 'to_k', 'to_v', 'to_out', 'proj', 'linear']
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        should_apply = any(target in name for target in target_modules)
        
        if should_apply and isinstance(module, nn.Linear):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            # Replace with LoRA version
            lora_layer = LinearWithLoRA(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
    
    return model


def inject_lora_to_diffusion_unet(diffusion_model, rank=4, alpha=1.0):
    """
    Inject LoRA specifically into UNet's attention layers
    This targets the most important layers for adaptation
    """
    # Target cross-attention and self-attention projection layers
    target_patterns = [
        'attention_1',  # Self attention
        'attention_2',  # Cross attention
        'q_proj',
        'k_proj', 
        'v_proj',
        'out_proj',
        'in_proj'
    ]
    
    lora_params = []
    
    for name, module in diffusion_model.named_modules():
        # Check if this is an attention layer
        is_attention_layer = any(pattern in name for pattern in target_patterns)
        
        if is_attention_layer and isinstance(module, nn.Linear):
            # Get parent module path
            parts = name.split('.')
            parent_path = '.'.join(parts[:-1])
            child_name = parts[-1]
            
            # Get parent module
            if parent_path:
                parent = diffusion_model.get_submodule(parent_path)
            else:
                parent = diffusion_model
            
            # Replace with LoRA version
            lora_layer = LinearWithLoRA(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            
            # Collect LoRA parameters
            lora_params.extend(lora_layer.lora.parameters())
            
            print(f"Applied LoRA to: {name}")
    
    return diffusion_model, lora_params


def get_lora_parameters(model):
    """Extract only LoRA parameters from the model"""
    lora_params = []
    for module in model.modules():
        if isinstance(module, (LinearWithLoRA, Conv2dWithLoRA)):
            lora_params.extend(module.lora.parameters())
    return lora_params


def save_lora_weights(model, path):
    """Save only LoRA weights (much smaller than full model)"""
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (LinearWithLoRA, Conv2dWithLoRA)):
            lora_state_dict[name + '.lora_A'] = module.lora.lora_A
            lora_state_dict[name + '.lora_B'] = module.lora.lora_B
    
    torch.save(lora_state_dict, path)
    print(f"LoRA weights saved to {path}")
    
    # Print size comparison
    import os
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"LoRA checkpoint size: {size_mb:.2f} MB")


def load_lora_weights(model, path):
    """Load LoRA weights into model"""
    lora_state_dict = torch.load(path)
    
    for name, module in model.named_modules():
        if isinstance(module, (LinearWithLoRA, Conv2dWithLoRA)):
            if name + '.lora_A' in lora_state_dict:
                module.lora.lora_A.data = lora_state_dict[name + '.lora_A']
                module.lora.lora_B.data = lora_state_dict[name + '.lora_B']
    
    print(f"LoRA weights loaded from {path}")
    return model