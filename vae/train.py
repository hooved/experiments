# minimal vae training implementation
from torch import nn, Tensor
from torch.nn.functional import silu, pad
import helpers
ModuleList = helpers.ModuleListTyped
Conv2d, GroupNorm = helpers.Conv2dTyped, helpers.GroupNormTyped

class ResnetBlock(nn.Module):
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

class Downsample(nn.Module):
  def __init__(self, ch:int):
    super().__init__()
    self.conv = Conv2d(ch, ch, kernel_size=3, stride=2, padding=0)

  def forward(self, x:Tensor) -> Tensor:
    x = pad(x, (0,1,0,1), mode="constant", value=0)
    x = self.conv(x)
    return x

class Down(nn.Module):
  def __init__(self, in_ch:int, out_ch:int, downsample:bool):
    super().__init__()
    self.block: ModuleList[ResnetBlock] = ModuleList([ResnetBlock(in_ch, out_ch) for _ in range(2)])
    if downsample:
      self.downsample = Downsample(in_ch)

  def forward(self, x:Tensor) -> Tensor:
    for resnetblock in self.block:
      x = resnetblock(x)
    if hasattr(self, "downsample"):
      x = self.downsample(x)
    return x

class AttnBlock(nn.Module):
  def __init__(self, ch:int):
    super().__init__()
    self.norm = GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)
    self.q = Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
    self.k = Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
    self.v = Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
    self.proj_out = Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)

class Middle(nn.Module):
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

class Encoder(nn.Module):
  def __init__(self, in_ch:int, ch:int):
    super().__init__()
    self.conv_in = Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
    self.down: ModuleList[Down] = ModuleList()
    for in_mult, out_mult, downsample in zip((1, 1, 2, 4), (1, 2, 4, 4), (False, False, False, True)):
      self.down.append(Down(in_mult*ch, out_mult*ch, downsample))

    self.mid = Middle(in_mult*ch)
    self.norm_out = GroupNorm(num_groups=32, num_channels=in_mult*ch, eps=1e-6, affine=True)
    self.conv_out = Conv2d(in_mult*ch, 8, kernel_size=3, stride=1, padding=1)

  def forward(self, x:Tensor) -> Tensor:
    for level in range(4):
      x = self.down[level](x)
    
    x = self.mid(x)
    x = self.norm_out(x)
    x = silu(x)
    x = self.conv_out(x)
    return x