import torch
import torch.nn as nn 
import torch.nn.functional as F
from . import clip
from . import model


from .simple_tokenizer import SimpleTokenizer as _Tokenizer
import os


def load_clip_to_cpu(backbone_name, pretrained_dir=None):
    if pretrained_dir is not None:
        state_dict = torch.load(pretrained_dir, map_location="cpu")

    else:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.clip_model = clip_model
        self.clip_model.requires_grad_(False)

        self.visiontransformer = self.clip_model.visual


    def forward(self, x, mask, mask_embedding, pos_embedding=True, return_token=False):
        x = self.visiontransformer.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        if mask is not None:
            x = x * mask + mask_embedding[0].unsqueeze(0) * (1 - mask)

        x = torch.cat([self.visiontransformer.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        if pos_embedding:
            positional_embedding_resize = F.interpolate(self.visiontransformer.positional_embedding.unsqueeze(0).unsqueeze(0), size=(x.size(1), x.size(2)), mode='bicubic').squeeze(0).squeeze(0)
            x = x + positional_embedding_resize.to(x.dtype)

        x = self.visiontransformer.ln_pre(x)
        x = x.permute(1, 0, 2)

       
        x = self.visiontransformer.transformer(x)
        x = x.permute(1, 0, 2)

        token = self.visiontransformer.ln_post(x[:, 1:, :])  

        x = self.visiontransformer.ln_post(x[:, 0, :])

        if self.visiontransformer.proj is not None:
            x = x @ self.visiontransformer.proj

        if return_token:
            return x, token
        else:
            return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="", cfg_imsize=224, class_specify=False, class_token_position='middle'):
        super().__init__()

        self.classnames = classnames
        self.n_cls = len(classnames)
        self._tokenizer = _Tokenizer()
        self.clip_model = clip_model
        self.clip_model.requires_grad_(False)

        self.dtype = self.clip_model.dtype
        self.ctx_dim = self.clip_model.ln_final.weight.shape[0]
        clip_imsize = self.clip_model.visual.input_resolution
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.ctx_init = ctx_init
        self.class_specify = class_specify
        self.class_token_position = class_token_position
        
        
        self.ctx = None
        self.n_ctx = 16
        self.tokenized_prompts = None  
        self.name_lens = None

        if self.ctx_init:
            self.ctx_init = self.ctx_init.replace("_", " ")
            self.n_ctx = len(self.ctx_init.split(" "))
            prompt = clip.tokenize(self.ctx_init)
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + self.n_ctx, :]
            self.prompt_prefix = self.ctx_init

        else:
            if self.class_specify:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(self.n_cls, self.n_ctx, self.ctx_dim, dtype=self.dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.prompt_prefix = " ".join(["X"] * self.n_ctx)


        self.ctx = nn.Parameter(ctx_vectors, requires_grad=True)  
        

    def forward(self, position_name="", device=None):


        
        self.classnames = [name.replace("_", " ") for name in self.classnames]
        name_lens = [len(self._tokenizer.encode(name)) for name in self.classnames]

        position_name_len = len(self._tokenizer.encode(position_name)) 
        prompts = [position_name + " " + self.prompt_prefix + " " + name + "." for name in self.classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
  
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts)
            embedding = embedding.to(dtype=self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1+position_name_len, :])  
        self.register_buffer("token_suffix", embedding[:, 1+position_name_len + self.n_ctx :, :])  


        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
        

        
        if self.clip_model.training:
            self.clip_model.eval()

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, tokenized_prompts


class ImageMaskLearner(nn.Module):
    def __init__(self, input_resolution, patch_size, width, dtype, mask_emb_depth=1):
        super().__init__()

        self.dtype = dtype
        self.input_resolution = input_resolution
        self.mask_pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        self.mask_prompt_depth = mask_emb_depth

    
        mask_vectors = torch.empty(self.mask_prompt_depth, self.input_resolution // patch_size * self.input_resolution // patch_size, width, dtype=self.dtype)
        nn.init.normal_(mask_vectors, std=0.02)
        self.mask_embedding = nn.Parameter(mask_vectors, requires_grad=True)

        self.mask_embedding.data *= 10


    def init_mask_embedding(self, mode="mae"):
        if mode == 'mae':
            torch.nn.init.normal_(self.mask_embedding, std=.02)
        elif mode == 'vpt_random':
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.width)) 
            nn.init.uniform_(self.mask_embedding, -val, val)
        elif mode == 'zeros':
            pass

    def forward(self, mask):
        mask = self.mask_pool(mask.squeeze(1)).reshape(mask.shape[0], -1).unsqueeze(-1)
        mask = torch.ceil(mask)

        if self.mask_embedding.shape[1] == 1:
            mask_embedding = self.mask_embedding.repeat(1, mask.shape[1], 1)

        else:
            mask_embedding = self.mask_embedding

        return mask.to(dtype=self.dtype), mask_embedding.to(dtype=self.dtype) 


class CustomCLIP(nn.Module):
    def __init__(self, classnames, backbone_name='ViT-B/32', n_ctx=16, ctx_init="", cfg_imsize=224, class_specify=False, class_token_position='middle', pretrained_dir=None):
        super().__init__()

        clip_model = load_clip_to_cpu(backbone_name, pretrained_dir)        
        self.dtype = clip_model.dtype

        self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx=n_ctx, ctx_init=ctx_init, cfg_imsize=cfg_imsize, class_specify=class_specify, class_token_position=class_token_position)
        self.image_encoder = ImageEncoder(clip_model)
        self.image_mask_learner = ImageMaskLearner(clip_model.image_resolution, clip_model.vision_patch_size, clip_model.vision_width, self.dtype)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale


    def forward(self, image, pos_embedding=False, return_token=False, image_mask=None, position_name=""):
        if self.image_encoder.training:
            self.image_encoder.eval()
            self.text_encoder.eval()
        if image_mask is not None:
            mask, mask_embedding = self.image_mask_learner(image_mask)
        else:
            mask = None
            mask_embedding = None
        if return_token:
            image_features, token_features = self.image_encoder(image.type(self.dtype), return_token=return_token, pos_embedding=pos_embedding, mask=mask, mask_embedding=mask_embedding)
        else:
            image_features = self.image_encoder(image.type(self.dtype), return_token=return_token, pos_embedding=pos_embedding, mask=mask, mask_embedding=mask_embedding)
        
        
        device = image.device

        logits = []
        for i in range(len(position_name)):
            p_name = position_name[i] 
            prompts, tokenized_prompts = self.prompt_learner(position_name=p_name, device=device)
            text_feature = self.text_encoder(prompts, tokenized_prompts)
            image_feature = image_features[i]

            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            logit = self.logit_scale.exp() * image_feature @ text_feature.t()
            logits.append(logit)


        logits = torch.stack(logits, dim=0)

        return logits



class PPAD(nn.Module):
    def __init__(self, classnames, backbone_name='ViT-B/32', n_ctx=16, ctx_init="", cfg_imsize=224, class_specify=False, class_token_position='middle', pretrained_dir=None, pos_embedding=False, return_tokens=False):
        super().__init__()
        self.num_clip = len(classnames)
        self.classnames = classnames 
        self.customclip = CustomCLIP(classnames=classnames,
                              backbone_name=backbone_name,
                              n_ctx=n_ctx, ctx_init=ctx_init,
                              cfg_imsize=cfg_imsize,
                              class_specify=class_specify,
                              class_token_position=class_token_position,
                              pretrained_dir=pretrained_dir)
        self.pos_embedding = pos_embedding
        self.return_tokens = return_tokens

    def forward(self, image, mask=None, position_name=""):
        logits = self.customclip(image, pos_embedding=self.pos_embedding, return_token=self.return_tokens, image_mask=mask, position_name=position_name)
        return logits
