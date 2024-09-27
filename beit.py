from torchvision.datasets.folder import default_loader
from torchvision import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
import numpy as np

from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig
# import utils

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
        checkpoint_activations=checkpoint_activations, 
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations, 
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class BEiT3ForRetrieval(BEiT3Wrapper):
    def __init__(
            self, 
            args,
            **kwargs
    ):
        super(BEiT3ForRetrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.criterion = None
#         self.criterion = utils.ClipLoss(
#             rank=utils.get_rank(), 
#             world_size=utils.get_world_size(), 
#         )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text_description=None, padding_mask=None, only_infer=False, **kwargs):
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None, 
                visual_tokens=image, 
                text_padding_position=None, 
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)
        else:
            vision_cls = None

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description, 
                visual_tokens=None, 
                text_padding_position=padding_mask, 
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None
        
        if only_infer:
            return vision_cls, language_cls
        else:
            loss, logits_per_image, logits_per_text = self.criterion(
                vision_cls, language_cls, self.logit_scale.exp())
            return loss, vision_cls, language_cls


@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model

@register_model
def beit3_base_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model

def build_transform(input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return transform

def to_image_tokens(image_paths, device):
    batch = []
    for image_path in image_paths:
        transform = build_transform(384)
        image = default_loader(image_path)
        image = transform(image)
        batch.append(image)
#     image = image.unsqueeze(0)
    return batch

def calc_img_embedding(model, image_paths, device):
    img_tokens = to_image_tokens(image_paths, device)
    print(len(img_tokens))
    img_tokens_stacked = torch.stack(img_tokens, dim=0)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        img_embedding = model(image=img_tokens_stacked, only_infer=True)
    del img_tokens
    del img_tokens_stacked
    return img_embedding[0]

def get_sentencepiece_model_for_beit3(model_path):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(model_path)

def to_text_tokens(text, tokenizer, max_len = 64):

    tokens_orig = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens_orig)
    tokens = token_ids

    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]

    tokens = [tokenizer.bos_token_id] + tokens[:] + [tokenizer.eos_token_id]
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
    tokens_true = tokens + [tokenizer.pad_token_id] * (max_len - num_tokens)

    padding_mask_tensor = torch.tensor(padding_mask).reshape(1, -1)
    token_ids_tensor = torch.tensor(tokens_true).reshape(1, -1)
    
    return token_ids_tensor, padding_mask_tensor

def calc_text_embedding(text, tokenizer):
    text_tokens, padding_mask = to_text_tokens(text, tokenizer)
    text_embedding = model(text_description=text_tokens, padding_mask=padding_mask, only_infer=True)
    return text_embedding[1]

import torch
import timm

model_name = "beit3_base_patch16_384_retrieval"  # Replace with the specific model you want to use, e.g., "resnet50", "efficientnet_b3", etc.
num_classes = 10  # Replace with the number of output classes in your model

models = timm.models.create_model(model_name, pretrained=False, num_classes=num_classes)

ckpt_path = "/kaggle/input/beit3-large-for-retrieval/beit3_base_patch16_384_f30k_retrieval.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(ckpt_path, map_location=device)
# print(checkpoint)

# Step 4: Load the model weights from the checkpoint
models.load_state_dict(checkpoint['model'])

import gc
import glob

import os 
batch_size = 512
file_data = ''
image_dir = [x for x in glob.glob('/root/data/0102/Keyframes')]
output_dir = f'/root/beit3'
os.makedirs(output_dir, exist_ok=True)
from tqdm import tqdm
# file_list = 
# subsubs = sorted(os.listdir(image_dir)) # ['L23_extra', 'L24_extra']

for folder in tqdm(image_dir):
    subsubs = sorted(glob.glob(folder+'/*'))
    for subsub in tqdm(subsubs):
        for v in sorted(glob.glob(f'{subsub}/*')):
            keyframe_paths = sorted(glob.glob(f'{v}/*'))
            image_embeddings = []
            path = v.replace('_extra','')
            data_part = v[-14:-11]
            video_id = v[-4:]
            output_filename = f"{data_part}/{video_id}.npy"
            output_path = os.path.join(output_dir, output_filename)
    #         output_filename = output_filename.replace("_extra", "")
            os.makedirs(os.path.join(output_dir, data_part), exist_ok=True)
            
            video_feats = []
            for batch_start in tqdm(range(0, len(keyframe_paths), batch_size)):
                batch_paths = keyframe_paths[batch_start:batch_start + batch_size]
                batch_images = [image_path for image_path in batch_paths]

                embeddings = calc_img_embedding(models, batch_images, device)
                print(embeddings)
                for b in range(embeddings.shape[0]):
                    video_feats.append(embeddings[b].cpu().numpy().astype(np.float32).flatten())

                del embeddings
                image_embeddings = []
                gc.collect()
            np.save(output_filename, video_feats)