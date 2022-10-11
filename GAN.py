



# The generator Class
class Generator(nn.Module):

    def __init__(self, base_channels=64, z_dim=128, channel=3, n_classes=2, upsample = True):
        super().__init__()
        
        self.z_dim = z_dim
 
        self.proj_z = SNLinear(z_dim, 1024 * 4 ** 2)
 
        self.g_blocks = nn.ModuleList([
                nn.ModuleList([
                    GResidualBlock( 16 * base_channels, 16 * base_channels),
                ]),
                
                nn.ModuleList([
                    GResidualBlock( 16 * base_channels, 8 * base_channels),
                ]),
                
                nn.ModuleList([
                    GResidualBlock( 8 * base_channels, 8 * base_channels),
                    CBAM(base_channels),

                ]),

                nn.ModuleList([
                    GResidualBlock(8 * base_channels, 4*base_channels),
                    CBAM(base_channels),

                ]),
            
                nn.ModuleList([
                    GResidualBlock(4 * base_channels, 2* base_channels),
                    CBAM(base_channels),

                ]),               
                 
                nn.ModuleList([
                    GResidualBlock(2 * base_channels, 1* base_channels),
                    CBAM(base_channels)
                ])
           
        ])
        
        # Using non-spectral Norm
        self.proj_o = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, channel, kernel_size=3, padding=1, stride=1)
        )
 
    def forward(self, z):
        '''
        z: random noise with size self.z_dim
        y: class embeddings with size self.shared_dim
            = NOTE =
            y should be class embeddings from self.shared_emb, not the raw class labels
        '''
        #First Linear Layer
        h = self.proj_z(z)
        
        h = h.view(h.size(0),-1, 4, 4)
        
        # Loop over blocks
        for index, blocklist in enumerate(self.g_blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h)

        h = self.proj_o(h)
 
        return   torch.tanh(h)


class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
    base_channels: the number of base channels, a scalar
    n_classes: the number of image classes, a scalar
    '''

    def __init__(self, base_channels=64, n_classes=2, channel =1):
        super().__init__()
        


        self.blocks = nn.Sequential(
                DResidualBlock(channel, base_channels, downsample=True, use_preactivation=False),
                DResidualBlock(1 * base_channels, 2 * base_channels, downsample=True, use_preactivation=True),
                DResidualBlock(2 * base_channels, 4 * base_channels, downsample=True, use_preactivation=True),
                DResidualBlock(4 * base_channels, 8 * base_channels, downsample=True, use_preactivation=True),
                DResidualBlock(8 * base_channels, 8 * base_channels, downsample=False, use_preactivation=False)
                DResidualBlock(8 * base_channels, 16 * base_channels, downsample=True, use_preactivation=True),
                DResidualBlock(16 * base_channels, 16 * base_channels, downsample=False, use_preactivation=True),

                nn.ReLU()
            
)
            


        self.activation = nn.LeakyReLU(0.2)

        self.linear = SNLinear(1024, 1 )

        
    def extract_features(self, x):
        h = self.blocks(x)
        
        # use global sum pooling as used in Spectral Normalization Code SN-GAN
        h = torch.sum(self.activation(h), [2,3])
        
        output = h.view(-1,1024)
        
        return output

    def forward(self, x):
      h = self.extract_features(x)
    
      h = self.linear(h)
      h = h.view(-1)
      return h



