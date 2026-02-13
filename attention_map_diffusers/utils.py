import os

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0
)

from .modules import *


def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):

            timestep = module.processor.timestep

            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach \
                else module.processor.attn_map
            
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, JointAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, FluxAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_function(name))
    
    return model


def _safe_set_forward(module, new_forward, module_class):
    """
    Replace a module's forward method while preserving any accelerate
    CPU-offload hook that may already wrap it.

    accelerate's ``add_hook_to_module`` stores the *original* forward in
    ``module._old_forward`` and replaces ``module.forward`` with a wrapper
    that calls ``pre_forward`` (moves params to GPU) → ``_old_forward`` →
    ``post_forward`` (moves params back to CPU).

    If we blindly overwrite ``.forward`` we destroy that wrapper and the
    model never gets moved to GPU, causing device-mismatch errors.
    """
    bound = new_forward.__get__(module, module_class)
    if hasattr(module, '_hf_hook') and hasattr(module, '_old_forward'):
        # Accelerate hook is active – swap the *inner* forward it calls.
        module._old_forward = bound
    else:
        module.forward = bound


def replace_call_method_for_unet(model):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        from diffusers.models.unets import UNet2DConditionModel
        _safe_set_forward(model, UNet2DConditionModelForward, UNet2DConditionModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'Transformer2DModel':
            from diffusers.models import Transformer2DModel
            _safe_set_forward(layer, Transformer2DModelForward, Transformer2DModel)
        
        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            from diffusers.models.attention import BasicTransformerBlock
            _safe_set_forward(layer, BasicTransformerBlockForward, BasicTransformerBlock)
        
        replace_call_method_for_unet(layer)
    
    return model


# TODO: implement
# def replace_call_method_for_sana(model):
#     if model.__class__.__name__ == 'SanaTransformer2DModel':
#         from diffusers.models.transformers import SanaTransformer2DModel
#         _safe_set_forward(model, SanaTransformer2DModelForward, SanaTransformer2DModel)

#     for name, layer in model.named_children():
        
#         if layer.__class__.__name__ == 'SanaTransformerBlock':
#             from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
#             _safe_set_forward(layer, SanaTransformerBlockForward, SanaTransformerBlock)
        
#         replace_call_method_for_sana(layer)
    
#     return model


def replace_call_method_for_sd3(model):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        from diffusers.models.transformers import SD3Transformer2DModel
        _safe_set_forward(model, SD3Transformer2DModelForward, SD3Transformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'JointTransformerBlock':
            from diffusers.models.attention import JointTransformerBlock
            _safe_set_forward(layer, JointTransformerBlockForward, JointTransformerBlock)
        
        replace_call_method_for_sd3(layer)
    
    return model


def replace_call_method_for_flux(model):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        from diffusers.models.transformers import FluxTransformer2DModel
        _safe_set_forward(model, FluxTransformer2DModelForward, FluxTransformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'FluxTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
            _safe_set_forward(layer, FluxTransformerBlockForward, FluxTransformerBlock)
        
        replace_call_method_for_flux(layer)
    
    return model


def init_pipeline(pipeline):
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0

    def _align_param_bias_dtype(module):
        """
        Ensure every module's bias has the same dtype as its weight.
        Uses in-place .data replacement so that the original nn.Parameter
        objects (tracked by accelerate's CPU-offload hooks) are preserved.
        """
        for m in module.modules():
            weight = getattr(m, "weight", None)
            bias = getattr(m, "bias", None)
            if weight is not None and bias is not None and bias.dtype != weight.dtype:
                m.bias.data = bias.data.to(weight.dtype)

    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            JointAttnProcessor2_0.__call__ = joint_attn_call2_0
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)
            _align_param_bias_dtype(pipeline.transformer)
        
        elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
            FluxPipeline.__call__ = FluxPipeline_call
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)
            _align_param_bias_dtype(pipeline.transformer)

        # TODO: implement
        # elif pipeline.transformer.__class__.__name__ == 'SanaTransformer2DModel':
        #     from diffusers import SanaPipeline
        #     SanaPipeline.__call__ == SanaPipeline_call
        #     pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn2')
        #     pipeline.transformer = replace_call_method_for_sana(pipeline.transformer)

    else:
        if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2')
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)
            _align_param_bias_dtype(pipeline.unet)


    return pipeline


def process_token(token, startofword):
    if '</w>' in token:
        token = token.replace('</w>', '')
        if startofword:
            token = '<' + token + '>'
        else:
            token = '-' + token + '>'
            startofword = True
    elif token not in ['<|startoftext|>', '<|endoftext|>']:
        if startofword:
            token = '<' + token + '-'
            startofword = False
        else:
            token = '-' + token + '-'
    return token, startofword


def save_attention_image(attn_map, tokens, batch_dir, to_pil):
    startofword = True
    for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
        token, startofword = process_token(token, startofword)
        to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))


def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='attn_maps', unconditional=True):
    to_pil = ToPILImage()
    
    token_ids = tokenizer(prompts)['input_ids']
    token_ids = token_ids if token_ids and isinstance(token_ids[0], list) else [token_ids]
    total_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]
    
    os.makedirs(base_dir, exist_ok=True)
    
    total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]  # (batch, height, width, attn_dim)
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0
    
    for timestep, layers in attn_maps.items():
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        os.makedirs(timestep_dir, exist_ok=True)
        
        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f'{layer}')
            os.makedirs(layer_dir, exist_ok=True)
            
            attn_map = attn_map.sum(1).squeeze(1).permute(0, 3, 1, 2)
            if unconditional:
                attn_map = attn_map.chunk(2)[1]
            
            resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1
            
            for batch, (tokens, attn) in enumerate(zip(total_tokens, attn_map)):
                batch_dir = os.path.join(layer_dir, f'batch-{batch}')
                os.makedirs(batch_dir, exist_ok=True)
                save_attention_image(attn, tokens, batch_dir, to_pil)
    
    total_attn_map /= total_attn_map_number
    for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
        batch_dir = os.path.join(base_dir, f'batch-{batch}')
        os.makedirs(batch_dir, exist_ok=True)
        save_attention_image(attn_map, tokens, batch_dir, to_pil)