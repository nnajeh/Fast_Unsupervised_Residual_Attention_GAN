


#### Generator Residual Blocks
class GResidualBlock(nn.Module):
    '''
    GResidualBlock Class
    Values:
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    '''

    def __init__(self,in_channels, out_channels, upsample = True):
        super().__init__()

        self.conv1 = SNConv2d(in_channels, out_channels, kernel_size=3, padding= 1)
        self.conv2 = SNConv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(inplace = False)
        self.upsample = upsample
        self.upsample_fn = nn.Upsample(scale_factor=2)   

        self.mixin = (in_channels != out_channels) or upsample
        if self.mixin:
            self.conv_mixin = SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)

            
    def forward(self, x):
        
        h = self.bn1(x)
        h = self.activation(h)
        h = self.upsample_fn(h)
        x = self.upsample_fn(x)
        h = self.conv1(h)

        h = self.bn2(h)
        h = self.activation(h)
        h = self.conv2(h)

        if self.mixin:
            x = self.conv_mixin(x)

        return h + x
      
 
### Discriminator Residual Blocks

class DResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=True, use_preactivation=False, hw = 64 ):
        super().__init__()

        self.conv1 = SNConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = SNConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.bn = nn.InstanceNorm2d(out_channels)


        self.activation = nn.ReLU()
       # self.activation = nn.LeakyReLU(0.2)
        self.use_preactivation = use_preactivation  # apply preactivation in all except first dblock

        self.downsample = downsample    # downsample occurs in all except last dblock
        
        if downsample:
            self.downsample_fn = nn.AvgPool2d(2)
        self.mixin = (in_channels != out_channels) or downsample
        
        if self.mixin:
            self.conv_mixin = SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def _residual(self, x):
        if self.use_preactivation:
            if self.mixin:
                x = self.conv_mixin(x)
            if self.downsample:
                x = self.downsample_fn(x)
        else:
            if self.downsample:
                x = self.downsample_fn(x)
            if self.mixin:
                x = self.conv_mixin(x)
        return x

    def forward(self, x):
        # Apply preactivation if applicable
        if self.use_preactivation:
            h = self.activation(x)
            #F.relu(x)
        else:
            h = x
        h = self.bn(h)
        h = self.activation(h)
        h = self.conv1(h)
        
        h = self.bn(h)
        h = self.activation(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = self.downsample_fn(h)

        return h + self._residual(x)
      
      
      
 
# # Discriminator Residual Block
# Residual block for the discriminator
class ResidualConv(nn.Module):
  def __init__(self, in_channels, out_channels,  preactivation=False,  downsample=None, bn=None):
    
    super(ResidualConv, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
   # self.hidden_channels = self.out_channels if wide else self.in_channels

    self.preactivation = preactivation
    self.activation = nn.ReLU()
    
    self.downsample = downsample
    self.bn = bn
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.downsample_fn = nn.AvgPool2d(2,2)
    
        
    # Conv layers
    self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True))
    
    self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1,  bias=True))

    
    self.learnable_sc = True if (in_channels != out_channels) or self.downsample else False
    if self.learnable_sc:
      self.conv_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
    
    
  def shortcut(self, x):
    
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample_fn(x)
        
    else:  
      if self.downsample:
        x = self.downsample_fn(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x
    
  def forward(self, x):
    if self.bn:
        x = self.bn1(x)
        
    if self.preactivation:
      h = F.relu(x)
    else:
      h = x    
    

    h = self.conv1(h)
    
    if self.bn:
        h = self.bn2(h)
    h = self.conv2(self.activation(h))
    
    if self.downsample:
      h = self.downsample_fn(h)     
        
    return h + self.shortcut(x)

