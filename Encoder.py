# # Encoder

class Encoder(nn.Module):
    def __init__(self, channel =3, filters=[64, 128, 256,512, 1024], z_dim=128):
        super(Encoder, self).__init__()

        self.input_layer = SNConv2d(channel, filters[0], 3,1,1)
        
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], preactivation=False, downsample=False, bn=False)
                
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], preactivation=True, downsample=True, bn=True)
        
        self.residual_conv_3 = ResidualConv(filters[2], filters[3],  preactivation=True, downsample=True, bn=True)
        
        self.residual_conv_4 = ResidualConv(filters[3], filters[3], preactivation=False, downsample=False, bn=False)

        self.residual_conv_5 = ResidualConv(filters[3], filters[4],  preactivation=True, downsample=True, bn=True)
        
        self.residual_conv_6 = ResidualConv(filters[4], filters[4],  preactivation=True, downsample=True, bn=True)
      
 
        self.linear =  SNLinear(filters[4]*4*4, z_dim)
        
        
    def forward(self, x):
        
        x1 = self.input_layer(x)
        
        x2 = self.residual_conv_1(x1)
        
        x3 = self.residual_conv_2(x2)
        
        x4 = self.residual_conv_3(x3)    
        
        x5 = self.residual_conv_4(x4)
        
        x6 = self.residual_conv_5(x5)
        
        x6 = self.residual_conv_6(x6)
  
        x6 = x6.view(x6.shape[0], -1)

        output = self.linear(x6)

        return torch.tanh(output) 
