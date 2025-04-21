import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import einops
from torchvision.models.vision_transformer import EncoderBlock
from .detr.models.transformer import Transformer
from .detr.models.detr_vae import build_ACT_model_and_optimizer

class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args['kl_weight']
        self.use_proprio = args['use_proprio']
        self.use_cam_pose = args['use_cam_pose']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, cam_config=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        assert image.size(2) == 3 or image.size(2) == 9, "Image must have 3 or 9 channels"
        image[:, :, :3] = normalize(image[:, :, :3])

        if not self.use_proprio:
            qpos = torch.zeros_like(qpos)

        if self.use_cam_pose:
            zeros = torch.zeros(qpos.size(0), qpos.size(1) - cam_config.size(1)).to(qpos)
            qpos = torch.cat([cam_config, zeros], dim=1)

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class ViTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.image_hw = (256, 256)
        self.patch_size = 16
        self.img_seq_length = (self.image_hw[0] // self.patch_size) * (self.image_hw[1] // self.patch_size)
        self.dropout = 0.1
        self.num_heads = 8
        self.num_layers = 6
        self.chunk_size = args['chunk_size']
        self.hidden_dim = args['hidden_dim']
        self.dim_feedforward = args['dim_feedforward']
        self.plucker_as_pe = args['plucker_as_pe']

        assert self.image_hw[0] % self.patch_size == 0 and self.image_hw[1] % self.patch_size == 0, \
            f"Image dimensions {self.image_hw[0]}x{self.image_hw[1]} must be divisible by patch size {self.patch_size}"

        self.conv_proj = nn.Conv2d(
            in_channels=3 if self.plucker_as_pe else 9,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.out_proj = nn.Linear(self.hidden_dim, args['action_dim'])

        if self.plucker_as_pe:
            self.emb_proj = nn.Linear(6, self.hidden_dim)
            self.pe = nn.Parameter(torch.empty(1,
                                    self.chunk_size,
                                    self.hidden_dim).normal_(std=1))
        else:
            self.pe = nn.Parameter(torch.empty(1,
                                    self.chunk_size + self.img_seq_length,
                                    self.hidden_dim).normal_(std=1))

        self.dropout = nn.Dropout(self.dropout)
        self.layers = nn.ModuleDict()
        for i in range(self.num_layers):
            self.layers[f"encoder_layer_{i}"] = EncoderBlock(
                self.num_heads,
                self.hidden_dim,
                self.dim_feedforward,
                dropout=0.1,
                attention_dropout=0.1,
                norm_layer=nn.LayerNorm
            )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=args['lr'])

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, image):
        x = torch.zeros(image.size(0),
                        self.chunk_size + self.img_seq_length,
                        self.hidden_dim,
                        device=image.device)

        if self.plucker_as_pe:
            # get plucker embedding as positional encoding
            # image: (b, num_cam, c, h, w)
            plucker = image[:, :, 3:]
            plucker = plucker[:, :, :, self.patch_size//2::self.patch_size, self.patch_size//2::self.patch_size]
            plucker = einops.rearrange(plucker, 'b n c h w -> b (n h w) c')
            plucker = self.emb_proj(plucker) # (b, n, h)

            rgb = image[:, :, :3]
            rgb = einops.rearrange(rgb, 'b n c h w -> (b n) c h w')
            rgb = self.conv_proj(rgb)
            rgb = einops.rearrange(rgb, 'b h ph pw -> b (ph pw) h')

            x[:, :self.img_seq_length] = rgb + plucker
            # x[:, self.img_seq_length:] = self.pe
        else:
            image = einops.rearrange(image, 'b n c h w -> (b n) c h w')
            image = self.conv_proj(image)
            image = einops.rearrange(image, 'b h ph pw -> b (ph pw) h')
            x[:, :self.img_seq_length] = image
            # x = x + self.pe

        for layer in self.layers.values():
            if self.plucker_as_pe:
                # x[:, :self.img_seq_length] = x[:, :self.img_seq_length] + plucker
                x[:, self.img_seq_length:] = x[:, self.img_seq_length:] + self.pe
            else:
                x = x + self.pe
            x = layer(x)

        x = self.out_proj(x[:, self.img_seq_length:])

        return x

    def __call__(self, qpos, image, actions=None, is_pad=None, cam_config=None):
        if actions is not None:
            actions = actions[:, :self.chunk_size]
            a_hat = self.forward(image)
            all_mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = all_mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:
            a_hat = self.forward(image)
            return a_hat


class DecOnlyPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.image_hw = (256, 256)
        self.patch_size = 16
        self.img_seq_length = (self.image_hw[0] // self.patch_size) * (self.image_hw[1] // self.patch_size)
        self.plucker_as_pe = args['plucker_as_pe']
        self.hidden_dim = args['hidden_dim']
        self.chunk_size = args['chunk_size']

        self.transformer = Transformer(
            d_model=self.hidden_dim,
            dropout=args['dropout'],
            nhead=args['nheads'],
            dim_feedforward=args['dim_feedforward'],
            num_encoder_layers=args['enc_layers'],
            num_decoder_layers=args['dec_layers'],
            return_intermediate_dec=False,
            normalize_before=args['pre_norm']
        )
        self.conv_proj = nn.Conv2d(
            in_channels=3 if self.plucker_as_pe else 9,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.out_proj = nn.Linear(self.hidden_dim, args['action_dim'])
        self.emb_proj = nn.Linear(6, self.hidden_dim)
        self.query_embed = nn.Embedding(self.chunk_size, self.hidden_dim)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=args['lr'])

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, image):
        if self.plucker_as_pe:
            # get plucker embedding as positional encoding
            # image: (b, num_cam, c, h, w)
            plucker = image[:, :, 3:]
            plucker = plucker[:, :, :, self.patch_size//2::self.patch_size, self.patch_size//2::self.patch_size]
            ph, pw = plucker.size(3), plucker.size(4)
            plucker = einops.rearrange(plucker, 'b n c h w -> b (n h w) c')
            plucker = self.emb_proj(plucker) # (b, n, h)
            plucker = einops.rearrange(plucker, 'b (h w) c -> b c h w', h=ph, w=pw)

            rgb = image[:, :, :3]
            rgb = einops.rearrange(rgb, 'b n c h w -> (b n) c h w')
            rgb = self.conv_proj(rgb)
            x = self.transformer(rgb, None, self.query_embed.weight, plucker)[0]
        else:
            raise NotImplementedError
        x = self.out_proj(x)
        return x

    def __call__(self, qpos, image, actions=None, is_pad=None, cam_config=None):
        if actions is not None:
            actions = actions[:, :self.chunk_size]
            is_pad = is_pad[:, :self.chunk_size]
            a_hat = self.forward(image)
            # loss = F.mse_loss(actions, a_hat, reduction='none')
            loss = F.l1_loss(actions, a_hat, reduction='none')
            loss = (loss * ~is_pad.unsqueeze(-1)).mean()
            loss_dict = dict()
            loss_dict['loss'] = loss
            return loss_dict
        else:
            a_hat = self.forward(image)
            return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
