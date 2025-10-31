# minimal vae training implementation, based on https://github.com/CompVis/latent-diffusion
import torch, helpers, random, numpy as np
from torch import Tensor, nn, distributed as dist
from torch.nn.functional import silu, pad, scaled_dot_product_attention, interpolate, normalize
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2 as T
from torchvision.io import decode_image, ImageReadMode
from torchvision.models.vgg import vgg16
from helpers import ModuleListTyped as ModuleList, getenv, register_buffer, dl_cache
from pathlib import Path
from typing import Literal, Callable

Module, Sequential = helpers.ModuleTyped, helpers.SequentialTyped
Conv2d, GroupNorm = helpers.Conv2dTyped, helpers.GroupNormTyped
BatchNorm2d, LeakyReLU = nn.BatchNorm2d, nn.LeakyReLU

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

  def __call__(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return super().__call__(x)

class ScalingLayer(Module):
  def __init__(self):
    super().__init__()
    self.shift = register_buffer(self, "shift", Tensor([-0.030, -0.088, -0.188]).reshape(1,3,1,1))
    self.scale = register_buffer(self, "scale", Tensor([0.458, 0.448, 0.450]).reshape(1,3,1,1))
  
  def forward(self, x:Tensor) -> Tensor:
    return (x - self.shift) / self.scale

# based on merging torchvision/models/vgg.py make_layers function, together with ldm vgg16 module
class vgg16(nn.Module):
  def __init__(self):
    super().__init__()
    in_ch = 3
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    layers = []
    for v in cfg:
      if v == "M": layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      else:
        assert isinstance(v, int)
        layers += [Conv2d(in_ch, v, kernel_size=3, padding=1)]
        layers += [nn.ReLU(inplace=True)]
        in_ch = v
    self.features = Sequential(*layers)
    state_dict = torch.load(dl_cache("https://download.pytorch.org/models/vgg16-397923af.pth", "vgg16-397923af.pth"), map_location="cpu")
    state_dict = {k:v for k,v in state_dict.items() if not k.startswith("classifier")}
    self.load_state_dict(state_dict)

  def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    relu_1_2 = x = self.features[0:4](x)
    relu_2_2 = x = self.features[4:9](x)
    relu_3_3 = x = self.features[9:16](x)
    relu_4_3 = x = self.features[16:23](x)
    relu_5_3 = x = self.features[23:30](x)
    return [normalize(x, p=2, dim=1, eps=1e-10) for x in [relu_1_2, relu_2_2, relu_3_3, relu_4_3, relu_5_3]]

class NetLinLayer(Module):
  def __init__(self, ch_in:int, ch_out:int=1):
    super().__init__()
    self.model = Sequential(nn.Dropout(), Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False))

class LPIPS(nn.Module):
  def __init__(self):
    super().__init__()
    self.scaling_layer = ScalingLayer()
    self.net = vgg16()
    self.lin0 = NetLinLayer(64)
    self.lin1 = NetLinLayer(128)
    self.lin2 = NetLinLayer(256)
    self.lin3 = NetLinLayer(512)
    self.lin4 = NetLinLayer(512)
    self.load_state_dict(torch.load(dl_cache("https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1", "lpips.pth"), map_location="cpu"),
                         strict=False)
    for param in self.parameters():
      param.requires_grad=False

  def forward(self, original:Tensor, recon:Tensor) -> Tensor:
    original_vgg = self.net(self.scaling_layer(original))
    recon_vgg = self.net(self.scaling_layer(recon))
    ret = torch.zeros(original.shape[0], 1, 1, 1, device=original.device)
    for i, lin in enumerate([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]):
      ret += lin.model((original_vgg[i] - recon_vgg[i]) ** 2).mean(dim=[2,3], keepdim=True)
    return ret
  
  def __call__(self, original:Tensor, recon:Tensor) -> Tensor:
    return super().__call__(original, recon)

class NLayerDiscriminator(Module):
  def __init__(self, in_ch=3, ch=64, n_middle_layers=3):
    super().__init__()
    kw, pw = 4, 1
    mults = [1, 2, 4, 8] + [2**n for n in range(4, n_middle_layers+1)]
    layers = [Conv2d(in_ch, ch, kernel_size=kw, stride=2, padding=pw)]
    layers += [LeakyReLU(0.2, inplace=True)]
    for i in range(1, n_middle_layers+1):
      prev_ch = mults[i-1] * ch
      next_ch = mults[i] * ch
      stride = 2 if i < n_middle_layers else 1
      layers += [Conv2d(prev_ch, next_ch, kernel_size=kw, stride=stride, padding=pw, bias=False)]
      layers += [BatchNorm2d(next_ch)]
      layers += [LeakyReLU(0.2, inplace=True)]
    layers += [Conv2d(next_ch, 1, kernel_size=kw, stride=1, padding=pw)]
    
    for layer in layers:
      if isinstance(layer, Conv2d):
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
      elif isinstance(layer, BatchNorm2d):
        nn.init.normal_(layer.weight.data, 1.0, 0.02)

    self.main = Sequential(*layers)

  def forward(self, x:Tensor) -> Tensor:
    return self.main(x)
  
class TrainStep(nn.Module):
  def __init__(self, kl_weight=1.0e-6):
    self.ae = AutoencoderKL()
    self.logvar = nn.Parameter(torch.zeros(size=()))
    self.perceptual_loss = LPIPS().eval()
    self.discriminator = NLayerDiscriminator(in_ch=3, ch=64, n_middle_layers=3)
    self.kl_weight = kl_weight

  def forward(self, original:Tensor, use_discriminator=False) -> tuple[Tensor, Tensor|None]:
    recon, latent_mean, latent_logvar = self.ae(original)
    loss_recon = torch.abs(recon - original) + self.perceptual_loss(original, recon)
    loss_recon = torch.sum(loss_recon / torch.exp(self.logvar) + self.logvar) / loss_recon.shape[0]
    loss_kl = 0.5 * torch.sum(latent_mean.pow(2) + torch.exp(latent_logvar) - 1 - latent_logvar, dim=[1,2,3])
    loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
    if use_discriminator:
      pass
    loss = loss_recon + self.kl_weight * loss_kl
    return loss, None


class ImageDataset(Dataset):
  def __init__(self, img_dir:str, transform:Callable|None=None, exts:set[str]={".jpeg"}):
    self.paths = sorted([p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts])
    self.transform = transform

  def __len__(self): return len(self.paths)

  def __getitem__(self, i:int) -> Tensor:
    img = decode_image(self.paths[i], mode=ImageReadMode.RGB)
    img = self.transform(img) if self.transform else img
    assert isinstance(img, Tensor)
    return img

def common_transforms(img:Tensor) -> Tensor:
  # note reference uses albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
  img = img.unsqueeze(0).to(torch.float32)
  img = interpolate(img, size=(MODEL_IMG_LEN, MODEL_IMG_LEN), mode="bicubic", align_corners=False, antialias=True).squeeze(0)
  img = (img/127.5 - 1.0).contiguous().clone()
  return img

MODEL_IMG_LEN=256
def transforms_train(img:Tensor, crop=False) -> Tensor:
  crop_len = int(min(img.shape[1:]) * np.random.uniform(0.5, 1.0))
  img = T.RandomCrop(crop_len)(img)
  return common_transforms(img)

def transforms_eval(img:Tensor) -> Tensor:
  img = img[:, 0:MODEL_IMG_LEN, 0:MODEL_IMG_LEN]
  return common_transforms(img)

# TODO: per-epoch seed reset for simpler resume replication
def set_seed(seed:int, set_torch=False):
  random.seed(seed)
  np.random.seed(seed)
  if set_torch: torch.manual_seed(seed)
  #torch.use_deterministic_algorithms(True)
  #torch.backends.cudnn.benchmark = False

def make_dl(ds, BS:int, shuffle=False, drop_last=False, num_workers=4) -> tuple[DataLoader, DistributedSampler]:
  def worker_init_fn(worker_id:int):
    set_seed(torch.initial_seed() % 2**32)

  sampler=DistributedSampler(ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=shuffle, drop_last=drop_last)
  return DataLoader(ds, batch_size=BS, sampler=sampler, num_workers=num_workers, pin_memory=True,
                    persistent_workers=True, worker_init_fn=worker_init_fn), sampler

def train():
  assert torch.cuda.is_available()
  dist.init_process_group(backend="nccl")
  assert torch.distributed.is_available() and torch.distributed.is_initialized() and (local_rank:=getenv("LOCAL_RANK", -1)) >= 0

  config = {}
  WORLD_SIZE    = config["WORLD_SIZE"]    = dist.get_world_size()
  BS            = config["BS"]            = getenv("BS", 6)
  EVAL_BS       = config["EVAL_BS"]       = getenv("EVAL_BS", 6)
  GRADACC_STEPS = config["GRADACC_STEPS"] = getenv("GRADACC_STEPS", 2)
  BASE_LR       = config["BASE_LR"]       = getenv("BASE_LR", 4.5e-6)
  KL_WEIGHT     = config["KL_WEIGHT"]     = getenv("KL_WEIGHT", 1.0e-6)
  TRAIN_IMG_DIR = config["TRAIN_IMG_DIR"] = getenv("TRAIN_IMG_DIR", "")
  EVAL_IMG_DIR  = config["EVAL_IMG_DIR"]  = getenv("EVAL_IMG_DIR", "")
  SEED          = config["SEED"]          = getenv("SEED", 12345) % 2**32
  assert all(v for v in config.values()), f"set these env vars: {[k for k,v in config.items() if not v]}"
  set_seed(SEED, set_torch=True)
  lr = WORLD_SIZE * BS * GRADACC_STEPS * BASE_LR

  torch.cuda.set_device(local_rank)
  device = torch.device(f"cuda:{local_rank}")

  data_train = ImageDataset(TRAIN_IMG_DIR, transforms_train)
  # TODO: change to drop_last=False, implement rank-based batch pad masking
  dl_train, sampler = make_dl(data_train, BS, drop_last=True)
  data_eval = ImageDataset(EVAL_IMG_DIR, transforms_eval)
  dl_eval, _ = make_dl(data_eval, EVAL_BS, drop_last=True, shuffle=False)
  train_step = TrainStep(kl_weight=KL_WEIGHT).to(local_rank)
  opt_ae = torch.optim.Adam(list(train_step.ae.parameters()) + [train_step.logvar], lr=lr, betas=(0.5, 0.9))
  opt_disc = torch.optim.Adam(train_step.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
  train_step = DDP(train_step, device_ids=[local_rank])

  num_epochs = 100
  for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    use_discriminator = True if num_epochs >= 5 else False
    for i, batch in enumerate(dl_train):
      loss_recon, loss_disc = train_step(batch, use_discriminator)
      assert isinstance(loss_recon, Tensor) and (isinstance(loss_disc, (Tensor, type(None))))

if __name__=="__main__":
  device="cuda:3"
  test = LPIPS().eval().to(device)
  test_in_1 = torch.rand(1,3,256,256).to(device)
  test_in_2 = torch.rand(1,3,256,256).to(device)
  out = test(test_in_1, test_in_2)
  train()