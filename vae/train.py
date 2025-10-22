# minimal vae training implementation, based on https://github.com/CompVis/latent-diffusion
import torch, torchvision, helpers, random, numpy as np
from torch import Tensor, nn, distributed as dist
from torch.nn.functional import silu, pad, scaled_dot_product_attention, interpolate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from helpers import ModuleListTyped as ModuleList, getenv
from pathlib import Path
from PIL import Image
from typing import Literal

Module = helpers.ModuleCallTyped
Conv2d, GroupNorm = helpers.Conv2dTyped, helpers.GroupNormTyped

class ResnetBlock(Module):
  def __init__(self, in_ch:int, out_ch:int):
    super().__init__()
    self.in_ch, self.out_ch = in_ch, out_ch
    self.norm1 = GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True)
    self.conv1 = Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
    self.norm2 = GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6, affine=True)
    self.conv2 = Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
    if in_ch != out_ch:
      self.nin_shortcut = Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

  def forward(self, x:Tensor) -> Tensor:
    h = self.norm1(x)
    h = silu(h)
    h = self.conv1(h)
    h = self.norm2(h)
    h = silu(h)
    h = self.conv2(h)
    if self.in_ch != self.out_ch:
      x = self.nin_shortcut(x)
    return x + h

class Downsample(Module):
  def __init__(self, ch:int):
    super().__init__()
    self.conv = Conv2d(ch, ch, kernel_size=3, stride=2, padding=0)

  def forward(self, x:Tensor) -> Tensor:
    x = pad(x, (0,1,0,1), mode="constant", value=0)
    x = self.conv(x)
    return x

class Upsample(Module):
  def __init__(self, ch:int):
    super().__init__()
    self.conv = Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)

  def forward(self, x:Tensor) -> Tensor:
    x = interpolate(x, scale_factor=2.0, mode="nearest")
    x = self.conv(x)
    return x

class Step(Module):
  def __init__(self, in_ch:int, out_ch:int, direction:Literal["down", "up", None], num_res_blocks:int):
    super().__init__()
    self.block: ModuleList[ResnetBlock] = ModuleList()
    for x,y in [(in_ch, out_ch)] + [(out_ch, out_ch)] * (num_res_blocks - 1):
      self.block.append(ResnetBlock(x, y))
    if direction == "down":
      self.downsample = Downsample(out_ch)
    elif direction == "up":
      self.upsample = Upsample(out_ch)

  def forward(self, x:Tensor) -> Tensor:
    for resnetblock in self.block:
      x = resnetblock(x)
    if hasattr(self, "downsample"):
      x = self.downsample(x)
    elif hasattr(self, "upsample"):
      x = self.upsample(x)
    return x

class AttnBlock(Module):
  def __init__(self, ch:int):
    super().__init__()
    self.norm = GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)
    self.q = Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
    self.k = Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
    self.v = Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
    self.proj_out = Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)

  def forward(self, x:Tensor) -> Tensor:
    y = self.norm(x)
    B, C, H, W = y.shape
    q = self.q(y).flatten(2).transpose(1,2).unsqueeze(1)
    k = self.k(y).flatten(2).transpose(1,2).unsqueeze(1)
    v = self.v(y).flatten(2).transpose(1,2).unsqueeze(1)
    y = scaled_dot_product_attention(q, k, v).squeeze(1)
    y = y.transpose(1,2).reshape(B, C, H, W)
    y = self.proj_out(y)
    x = x + y
    return x

class Middle(Module):
  def __init__(self, ch:int):
    super().__init__()
    self.block_1 = ResnetBlock(ch, ch)
    self.attn_1 = AttnBlock(ch)
    self.block_2 = ResnetBlock(ch, ch)

  def forward(self, x:Tensor) -> Tensor:
    x = self.block_1(x)
    x = self.attn_1(x)
    x = self.block_2(x)
    return x

class Encoder(Module):
  def __init__(self, in_ch:int, ch:int):
    super().__init__()
    self.conv_in = Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
    self.down: ModuleList[Step] = ModuleList()
    for in_mult, out_mult, direction in zip((1, 1, 2, 4), (1, 2, 4, 4), ("down", "down", "down", None)):
      self.down.append(Step(in_mult*ch, out_mult*ch, direction, 2))

    self.mid = Middle(in_mult*ch)
    self.norm_out = GroupNorm(num_groups=32, num_channels=in_mult*ch, eps=1e-6, affine=True)
    self.conv_out = Conv2d(in_mult*ch, 8, kernel_size=3, stride=1, padding=1)

  def forward(self, x:Tensor) -> Tensor:
    x = self.conv_in(x)
    for level in range(4):
      x = self.down[level](x)
    
    x = self.mid(x)
    x = self.norm_out(x)
    x = silu(x)
    x = self.conv_out(x)
    return x

class Decoder(Module):
  def __init__(self, in_ch:int, ch:int):
    super().__init__()
    self.conv_in = Conv2d(in_ch, 4*ch, kernel_size=3, stride=1, padding=1)
    self.mid = Middle(512)
    self.up: ModuleList[Step] = ModuleList()
    for in_mult, out_mult, direction in zip((4, 4, 2, 1), (4, 2, 1, 1), ("up", "up", "up", None)):
      self.up.append(Step(in_mult*ch, out_mult*ch, direction, 3))

    self.norm_out = GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)
    self.conv_out = Conv2d(ch, 3, kernel_size=3, stride=1, padding=1)

  def forward(self, x:Tensor) -> Tensor:
    x = self.conv_in(x)
    x = self.mid(x)
    for level in range(4):
      x = self.up[level](x)

    x = self.norm_out(x)
    x = silu(x)
    x = self.conv_out(x)
    return x

class DiagonalGaussian(Module):
  def __init__(self, mean_logvar:Tensor):
    self.mean, self.logvar = mean_logvar.chunk(2, dim=1)

class AutoencoderKL(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder(in_ch=3, ch=128)
    self.quant_conv = Conv2d(8, 8, kernel_size=1)
    self.post_quant_conv = Conv2d(4, 4, kernel_size=1)
    self.decoder = Decoder(in_ch=4, ch=128)

  def encode(self, x:Tensor) -> Tensor:
    x = self.encoder(x)
    mean_logvar = self.quant_conv(x)
    return mean_logvar

  def decode(self, z:Tensor) -> Tensor:
    z = self.post_quant_conv(z)
    x_recon = self.decoder(z)
    return x_recon

  def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mean_logvar = self.encode(x)
    mean, logvar = mean_logvar.chunk(2, dim=1)
    logvar = logvar.clamp(-30.0, 20.0)
    z = mean + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
    x_recon = self.decode(z)
    return x_recon, mean, logvar

class ImageDataset(Dataset):
  def __init__(self, img_dir:str, transforms:torchvision.transforms.Compose|None=None, exts:set[str]={".jpeg"}):
    self.paths = sorted([p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts])
    self.transforms = transforms

  def __len__(self): return len(self.paths)

  def __getitem__(self, i:int) -> Tensor:
    img = Image.open(self.paths[i])
    img = img.convert("RGB") if img.mode != "RGB" else img
    img = self.transforms(img) if self.transforms else img
    assert isinstance(img, Tensor)
    return img

def set_seed(seed:int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  #torch.use_deterministic_algorithms(True)
  #torch.backends.cudnn.benchmark = False

def make_dl(ds, BS:int, shuffle=False, drop_last=False, num_workers=4) -> tuple[DataLoader, DistributedSampler]:
  def worker_init_fn(worker_id:int):
    set_seed(torch.initial_seed() % 2**32)

  sampler=DistributedSampler(ds, shuffle=shuffle, drop_last=drop_last)
  return DataLoader(ds, batch_size=BS, shuffle=False, sampler=sampler, drop_last=drop_last, num_workers=num_workers,
                    pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn), sampler

def train():
  assert torch.cuda.is_available()
  assert torch.distributed.is_available() and torch.distributed.is_initialized() and (local_rank:=getenv("LOCAL_RANK", -1)) >= 0

  config = {}
  BS            = config["BS"]            = getenv("BS", 6)
  EVAL_BS       = config["EVAL_BS"]       = getenv("EVAL_BS", 6)
  TRAIN_IMG_DIR = config["TRAIN_IMG_DIR"] = getenv("TRAIN_IMG_DIR", "")
  EVAL_IMG_DIR  = config["EVAL_IMG_DIR"]  = getenv("EVAL_IMG_DIR", "")
  SEED          = config["SEED"]          = getenv("SEED", 12345) % 2**32
  assert all(v for v in config.values()), f"set these env vars: {[k for k,v in config.items() if not v]}"
  set_seed(SEED)

  dist.init_process_group(backend="nccl")
  torch.cuda.set_device(local_rank)
  device = torch.device(f"cuda:{local_rank}")

  data_train = ImageDataset(TRAIN_IMG_DIR)
  dl_train, sampler = make_dl(data_train, BS)
  data_eval = ImageDataset(EVAL_IMG_DIR)
  dl_eval, _ = make_dl(data_eval, EVAL_BS, shuffle=False)

if __name__=="__main__":
  train()