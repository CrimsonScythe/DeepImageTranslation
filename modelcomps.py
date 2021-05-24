import torch.nn as nn
import torch


# https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/21
# https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/ops/random_ops.py#L163


def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
      cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
      if not torch.sum(cond):
        break
      t = torch.where(cond.cuda(), torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std).cuda(), t.cuda()).cuda()
      
'''
Dynamically compute padding based on input
'''

def compute_conv_output(H, K, S):
  return int(((H-K+2*0)/S)+1)

def compute_padding(out, H, K, S):
  return int((S*(out-1)-H+K)/2)

def compute_padding_deconv(out, H, K, S):
  return int((-out+(H-1)*S+K)/2)

def compute_maxpool_padding(S, H, f):
  return int((S*(H-1)-H+f)/2)

#####################################################
# MODEL
#####################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class General_Conv2D(nn.Module):
    # @torch.no_grad()
    def init_weights(self, n):
      if type(n) == nn.Conv2d:
          truncated_normal(t=n.weight, std=0.01)
          

    def __init__(self, in_features=0, out_features=64, k=7, s=1, stddev=0.01, do_relu=True, keep_rate=None, relu_factor=0, norm_type=None, train=True, padding=False):
        super(General_Conv2D, self).__init__()
        self.std=stddev
        self.keep_rate=keep_rate
        self.relu_factor=relu_factor
        self.norm_type=norm_type
        self.do_relu=do_relu
        self.k=k
        self.s=s
        self.padding=padding
        self.stddev = stddev

        if not keep_rate is None:
          self.dropout = nn.Dropout(p=keep_rate, inplace=True)
        
        if norm_type=='Batch':
            self.batchnorm=nn.BatchNorm2d(num_features=out_features, momentum=0.1, affine=True)
        elif norm_type=='Ins':
            self.instancenorm=nn.InstanceNorm2d(num_features=out_features)
        
        self.relu=nn.ReLU()
        self.lrealu=nn.LeakyReLU()
        
        self.w= nn.Parameter(torch.randn(out_features, in_features, k,k))
        truncated_normal(t=self.w, std=self.stddev)
        self.b=nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        
        if self.padding==False:
          x = torch.nn.functional.conv2d(x, self.w, self.b, padding=0)

        else:
          x = torch.nn.functional.conv2d(x, self.w, self.b, padding=compute_padding(x.shape[-1], x.shape[-1], self.k, self.s))
        
        '''
        dropout
        '''
        if not self.keep_rate is None:
          x=self.dropout(x)

        '''
        norm
        '''
        if (self.norm_type=='Batch'):
          x=self.batchnorm(x)
        elif (self.norm_type=='Ins'):
          x=self.instancenorm(x)

        '''
        act fnc
        '''
        if (self.do_relu==True):
          if (self.relu_factor==0):
            x=self.relu(x)
          else:
            x=self.lrealu(x)
        
        return x

class Resnet_Block(nn.Module):
    def __init__(self, in_features, out_features , padding="REFLECT", norm_type=None, keep_rate=0.25):
        super(Resnet_Block, self).__init__()

        if padding=='REFLECT':
          pad_layer=nn.ReflectionPad2d(1)
        elif padding=='CONSTANT':
          pad_layer=nn.ZeroPad2d(1)
        else: # SYMMETRIC
          pad_layer=nn.ReplicationPad2d(1)

        self.block = nn.ModuleList([
            pad_layer,
            General_Conv2D(in_features, out_features, k=3, s=1, stddev=0.01, norm_type=norm_type, keep_rate=keep_rate, padding=False),
            pad_layer,
            General_Conv2D(in_features, out_features, k=3, s=1, stddev=0.01, do_relu=False, norm_type=norm_type, keep_rate=keep_rate, padding=False)
        ])

      
        self.relu=nn.ReLU()

    def forward(self, x):      
      
        self.block[1].dropout.train()
        self.block[3].dropout.train()

        for i in range(len(self.block)):
          if i==0:
            out = self.block[i](x)
          else:
            out = self.block[i](out)

  
        return self.relu(out+x)

class Resnet_Block_ds(nn.Module):
    def __init__(self, in_features=0, out_features=0, padding="REFLECT",dim=0, norm_type=None, keep_rate=None):
        super(Resnet_Block_ds, self).__init__()

        self.in_features=in_features
        self.out_features=out_features

        if padding=='REFLECT':
          pad_layer=nn.ReflectionPad2d(1)
        elif padding=='CONSTANT':
          pad_layer=nn.ZeroPad2d(1)
        else: # SYMMETRIC
          pad_layer=nn.ReplicationPad2d(1)


        self.block = nn.ModuleList([
            
            pad_layer,
            General_Conv2D(in_features, out_features, k=3, s=1, stddev=0.01, norm_type=norm_type, keep_rate=keep_rate, padding=False),
            pad_layer,
            General_Conv2D(out_features, out_features, k=3, s=1, stddev=0.01, do_relu=False, norm_type=norm_type, keep_rate=keep_rate, padding=False)
        ])

        self.relu = nn.ReLU()

    def forward(self, x):
     
        for i in range(len(self.block)):
          if i==0:
            out = self.block[i](x)
          else:
            out = self.block[i](out)
        '''
        pad channel dim
        '''
        pd=(self.out_features-self.in_features) // 2

        padding = torch.zeros(x.shape[0], pd, x.shape[2], x.shape[3]).cuda()
        padded_inp = torch.cat((x, padding), 1)
        padded_inp = torch.cat((padding, padded_inp), 1)


        return self.relu(out+padded_inp)
  
class drn_Block(nn.Module):
    def __init__(self, in_features):
        super(drn_Block, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_features, in_features, 3, dilation=2),
            nn.Dropout(0.0, inplace=True), # TODO remove hard coded dropout value
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_features, in_features, 3, dilation=2),
            nn.Dropout(0.0, inplace=True), # TODO remove hard coded dropout value
            nn.BatchNorm2d(in_features),
        )

        self.relu=nn.ReLU()

    def forward(self, x):
        x = x + self.block(x)
        
        return self.relu(x)

class General_Conv2D_GA(nn.Module):
    # @torch.no_grad()
    def init_weights(self, n):
      if type(n) == nn.Conv2d:
          truncated_normal(n.weight, 0.02)
          

    def __init__(self, in_features=0, out_features=64, k=7, s=1, stddev=0.02, do_relu=True, keep_rate=None, relu_factor=0, norm_type=None, train=True, padding=False, input_dim=0):
        super(General_Conv2D_GA, self).__init__()
        self.std=stddev
        self.keep_rate=keep_rate
        self.relu_factor=relu_factor
        self.norm_type=norm_type
        self.do_relu=do_relu
        self.k=k
        self.s=s
        self.padding=padding
        self.stddev=stddev

        if not keep_rate is None:
          self.dropout = nn.Dropout(p=keep_rate, inplace=True)
        
        if norm_type=='Batch':
          self.batchnorm=nn.BatchNorm2d(num_features=out_features, momentum=0.1, affine=True)
        elif norm_type=='Ins':
          self.instancenorm=nn.InstanceNorm2d(num_features=out_features)
        
        self.relu=nn.ReLU()
        self.lrealu=nn.LeakyReLU()
        
        self.w= nn.Parameter(torch.randn(out_features, in_features, k,k))
        truncated_normal(t=self.w, std=self.stddev)
        self.b=nn.Parameter(torch.randn(out_features))
        torch.nn.init.constant_(self.b, 0)

    def forward(self, x):

        '''
        initialize weights
        '''
     
        if self.padding==False:
          x = torch.nn.functional.conv2d(x, self.w, self.b, padding=0)

        else:
          
          x = torch.nn.functional.conv2d(x, self.w, self.b, padding=compute_padding(x.shape[-1], x.shape[-1], self.k, self.s))
        
        '''
        dropout
        '''
        if not self.keep_rate is None:
          x=self.dropout(x)

        '''
        norm
        '''
        if (self.norm_type=='Batch'):
          x=self.batchnorm(x)
        elif self.norm_type=='Ins':# Instance norm
          x=self.instancenorm(x)

        '''
        act fnc
        '''
        if (self.do_relu==True):
          if (self.relu_factor==0):
            x=self.relu(x)
          else:
            x=self.lrealu(x)
        
        return x

class Resnet_Block_ins(nn.Module):
    def __init__(self, in_features=0, out_features=0, padding="REFLECT",dim=0, norm_type=None):
        super(Resnet_Block_ins, self).__init__()

        self.in_features=in_features
        self.out_features=out_features

        if padding=='REFLECT':
          pad_layer=nn.ReflectionPad2d(1)
        elif padding=='CONSTANT':
          pad_layer=nn.ZeroPad2d(1)
        else: # SYMMETRIC
          pad_layer=nn.ReplicationPad2d(1)

        self.block = nn.ModuleList([
            
            pad_layer,
            General_Conv2D_GA(in_features, out_features, k=3, s=1, stddev=0.02, norm_type='Ins', padding=False, do_relu=True),
            pad_layer,
            General_Conv2D_GA(out_features, out_features, k=3, s=1, stddev=0.02, norm_type='Ins', padding=False, do_relu=False)

            ]
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
      y=x

      for i in range(len(self.block)):
        if i==0:
          out=self.block[i](x)
        else:
          out=self.block[i](out)
     
      return self.relu(out+y)

class General_Deconv2D(nn.Module):
    # @torch.no_grad()
    def init_weights(self, n):
      if type(n) == nn.Conv2d:
          truncated_normal(n.weight, 0.02)
          

    def __init__(self, in_features=0, out_features=64, k=7, s=1, stddev=0.02, do_relu=True, keep_rate=None, relu_factor=0, norm_type=None, train=True, padding=False, input_dim=0, out_shape=0):
        super(General_Deconv2D, self).__init__()
        self.std=stddev
        self.keep_rate=keep_rate
        self.relu_factor=relu_factor
        self.norm_type=norm_type
        self.do_relu=do_relu
        self.k=k
        self.s=s
        self.padding=padding
        self.out_shape=out_shape
        self.stddev=stddev
        
        if norm_type=='Batch':
          self.batchnorm=nn.BatchNorm2d(num_features=out_features, momentum=0.1, affine=True)
        elif norm_type=='Ins':
          self.instancenorm=nn.InstanceNorm2d(num_features=out_features)
        
        self.relu=nn.ReLU()
        self.lrealu=nn.LeakyReLU()
        
        self.w= nn.Parameter(torch.randn(in_features, out_features, k,k))
        truncated_normal(self.w, std=self.stddev)
        self.b=nn.Parameter(torch.randn(out_features))
        torch.nn.init.constant_(self.b, 0)

    def forward(self, x):

        if self.padding==False:
          x = torch.nn.functional.conv_transpose2d(x, self.w, bias=self.b, stride=self.s, padding=0)

        else:
          x = torch.nn.functional.conv_transpose2d(x, self.w, bias=self.b, stride=self.s, padding=compute_padding_deconv(self.out_shape, x.shape[-1], self.k, self.s))

        '''
        norm
        '''
        if (self.norm_type=='Batch'):
          x=self.batchnorm(x)
        else:# Instance norm
          x=self.instancenorm(x)

        '''
        act fnc
        '''
        if (self.do_relu==True):
          if (self.relu_factor==0):
            x=self.relu(x)
          else:
            x=self.lrealu(x)
        
        return x

class GeneratorResNet(nn.Module):
    def __init__(self, skip=True):
        super(GeneratorResNet, self).__init__()

        self.skip=skip
        self.max_features = 32*4 # used to be 32*4 but there were memory issues.
        
        self.block = nn.ModuleList([
            nn.ZeroPad2d(3),
            General_Conv2D_GA(in_features=1, out_features=32, k=7, s=1, stddev=0.02, do_relu=True, norm_type='Ins', train=True, padding=False),
            General_Conv2D_GA(in_features=32, out_features=32*2, k=3, s=2, stddev=0.02, do_relu=True, norm_type='Ins', train=True, padding=True),
            General_Conv2D_GA(in_features=32*2, out_features=self.max_features, k=3, s=2, stddev=0.02, do_relu=True, norm_type='Ins', train=True, padding=True),
            Resnet_Block_ins(in_features=self.max_features, out_features=self.max_features, padding='CONSTANT'),
            Resnet_Block_ins(in_features=self.max_features, out_features=self.max_features, padding='CONSTANT'),
            Resnet_Block_ins(in_features=self.max_features, out_features=self.max_features, padding='CONSTANT'),
            Resnet_Block_ins(in_features=self.max_features, out_features=self.max_features, padding='CONSTANT'),
            Resnet_Block_ins(in_features=self.max_features, out_features=self.max_features, padding='CONSTANT'),
            Resnet_Block_ins(in_features=self.max_features, out_features=self.max_features, padding='CONSTANT'),
            Resnet_Block_ins(in_features=self.max_features, out_features=self.max_features, padding='CONSTANT'),
            Resnet_Block_ins(in_features=self.max_features, out_features=self.max_features, padding='CONSTANT'),
            Resnet_Block_ins(in_features=self.max_features, out_features=self.max_features, padding='CONSTANT'),
            General_Deconv2D(in_features=self.max_features, out_features=32*2, k=3, s=2, stddev=0.02, norm_type='Ins', out_shape=32, padding=True),
            General_Deconv2D(in_features=32*2, out_features=32, k=3, s=2, stddev=0.02, norm_type='Ins', out_shape=64, padding=True),
            General_Conv2D_GA(in_features=32, out_features=1, k=7, s=1, stddev=0.02, do_relu=False, train=True, padding=True),
            ]
            )
        self.tanh = torch.nn.Tanh()
        self.pad = torch.nn.ReflectionPad2d((1,0,1,0))
        
        

    def forward(self, x):

      y=x

      for i in range(len(self.block)):
        if i==0:
          out=self.block[i](x)
        else:
          out=self.block[i](out)

      if self.skip==True:
        return self.tanh(out+y)
      else:
        return self.tanh(out)

'''
input shape: B,C,H,W
'''
class EncoderNet(nn.Module):
    def __init__(self, input_shape):
        super(EncoderNet, self).__init__()
        
        self.input_shape=input_shape

        self.out_c1 = General_Conv2D(in_features=1, out_features=16, padding=True, input_dim=self.input_shape[-1], norm_type='Batch', keep_rate=0.0)
        self.out_res1 = Resnet_Block(in_features=16,out_features=16,norm_type='Batch', keep_rate=0.0)
        self.out1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=compute_maxpool_padding(S=1, H=input_shape[-1], f=1))
        
        self.out_res2 = Resnet_Block_ds(in_features=16, out_features=16*2, norm_type='Batch', padding='CONSTANT', keep_rate=0.0)

        self.out_res3 = Resnet_Block_ds(in_features=16*2, out_features=16*4, norm_type='Batch', padding='CONSTANT', keep_rate=0.0)
        self.out_res4 = Resnet_Block(in_features=16*4,out_features=16*4,norm_type='Batch', padding='CONSTANT', keep_rate=0.0)

        self.out_res5 = Resnet_Block_ds(in_features=16*4,out_features=16*8,norm_type='Batch', padding='CONSTANT', keep_rate=0.0)
        self.out_res6 = Resnet_Block(in_features=16*8,out_features=16*8,norm_type='Batch', padding='CONSTANT', keep_rate=0.0)

        self.out_res7 = Resnet_Block_ds(in_features=16*8,out_features=16*16,norm_type='Batch', padding='CONSTANT', keep_rate=0.0)
        self.out_res8 = Resnet_Block(in_features=16*16,out_features=16*16,norm_type='Batch', padding='CONSTANT', keep_rate=0.0)

        self.out_res9 = Resnet_Block(in_features=16*16,out_features=16*16,norm_type='Batch', padding='CONSTANT', keep_rate=0.0)
        self.out_res10= Resnet_Block(in_features=16*16,out_features=16*16,norm_type='Batch', padding='CONSTANT', keep_rate=0.0)

        self.out_res11= Resnet_Block_ds(in_features=16*16,out_features=16*32,norm_type='Batch', padding='CONSTANT', keep_rate=0.0)
        self.out_res12= Resnet_Block(in_features=16*32,out_features=16*32,norm_type='Batch', padding='CONSTANT', keep_rate=0.0)

        self.out_drn1 = drn_Block(in_features=16*32, keep_rate=0.0)
        self.out_drn2 = drn_Block(in_features=16*32, keep_rate=0.0)
        
        self.out_c2 = General_Conv2D(in_features=16*32, out_features=16*32, k=3, padding=True, input_dim=0, norm_type='Batch', keep_rate=0.0)
        self.out_c3 = General_Conv2D(in_features=16*32, out_features=16*32, k=3, padding=True, input_dim=0, norm_type='Batch', keep_rate=0.0)


    def forward(self, x):
  
        '''
        TODO
        separate kernel size implementation from tf not available in pytorch
        '''
        x=self.out_c1(x)
        x=self.out_res1(x)
        x=self.out1(x)
        x=self.out_res2(x)
        x=torch.nn.functional.max_pool2d(x,kernel_size=1, stride=1, padding=compute_maxpool_padding(S=1, H=x.shape[-1], f=1))
        x=self.out_res3(x)
        x=self.out_res4(x)
        x=torch.nn.functional.max_pool2d(x,kernel_size=1, stride=1, padding=compute_maxpool_padding(S=1, H=x.shape[-1], f=1))
        x=self.out_res5(x)
        x=self.out_res6(x)
        x=self.out_res7(x)
        x=self.out_res8(x)
        x=self.out_res9(x)
        x=self.out_res10(x)
        x=self.out_res11(x)
        x=self.out_res12(x)
        y=x
        x=self.out_drn1(x)
        x=self.out_drn2(x)
        x=self.out_c2(x)
        x=self.out_c3(x)

        return x, y

class DecoderNet(nn.Module):
    def __init__(self, input_shape, skip=False):
        super(DecoderNet, self).__init__()
        
        self.skip=skip
        self.input_shape=input_shape
        
        self.block = nn.ModuleList([
          General_Conv2D(in_features=self.input_shape[1], k=3, s=1, out_features=32*4, padding=True, input_dim=self.input_shape[-1], norm_type='Ins'),
          Resnet_Block(in_features=32*4,out_features=32*4, padding='CONSTANT', norm_type='Ins', keep_rate=0.25),
          Resnet_Block(in_features=32*4,out_features=32*4,padding='CONSTANT',norm_type='Ins', keep_rate=0.25),
          Resnet_Block(in_features=32*4,out_features=32*4,padding='CONSTANT',norm_type='Ins', keep_rate=0.25),
          Resnet_Block(in_features=32*4,out_features=32*4,padding='CONSTANT',norm_type='Ins', keep_rate=0.25),
          General_Deconv2D(in_features=32*4, out_features=32*2, k=3, s=2, stddev=0.02, norm_type='Ins', out_shape=16, padding=True),
          General_Deconv2D(in_features=32*2, out_features=32*2, k=3, s=2, stddev=0.02, norm_type='Ins', out_shape=32, padding=True),
          General_Deconv2D(in_features=32*2, out_features=32, k=3, s=2, stddev=0.02, norm_type='Ins', out_shape=64, padding=True),
          General_Conv2D(in_features=32, out_features=1, k=7, s=1, norm_type=None, do_relu=False, stddev=0.02, padding=True, input_dim=self.input_shape[-1])
        ])

        self.tanh = nn.Tanh()

    def forward(self, x, input_img):
      for i in range(len(self.block)):
        if i==0:
          out=self.block[i](x)
        else:
          out=self.block[i](out)

      if self.skip==True:
        return self.tanh(input_img+out)
      else:
        return self.tanh(out)

class Discriminator(nn.Module):
    def __init__(self, input_shape, skip=False):
        super(Discriminator, self).__init__()
        
        self.skip=skip
        self.input_shape=input_shape

        self.block = nn.ModuleList([
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=input_shape[0], out_features=64, k=4, s=2, stddev=0.02, padding=False, relu_factor=0.02, input_dim=self.input_shape[-1], norm_type=None),
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=64, out_features=64*2, k=4, s=2, stddev=0.02, padding=False, relu_factor=0.02, input_dim=self.input_shape[-1], norm_type='Ins'),
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=64*2, out_features=64*4, k=4, s=2, stddev=0.02, padding=False, relu_factor=0.02, input_dim=self.input_shape[-1], norm_type='Ins'),
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=64*4, out_features=64*8, k=4, s=1, stddev=0.02, padding=False, relu_factor=0.02, input_dim=self.input_shape[-1], norm_type='Ins'),
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=64*8, out_features=1, k=4, s=1, stddev=0.02, padding=False, input_dim=self.input_shape[-1], norm_type=None, do_relu=False),
        ]
        )

    def forward(self, x):
      for i in range(len(self.block)):
          x=self.block[i](x)
      return x

class Discriminator_aux(nn.Module):
    def __init__(self, input_shape, skip=False):
        super(Discriminator_aux, self).__init__()
        
        self.skip=skip
        self.input_shape=input_shape

        self.block = nn.ModuleList([
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=1, out_features=64, k=4, s=2, stddev=0.02, padding=False, relu_factor=0.02, input_dim=self.input_shape[-1], norm_type=None),
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=64, out_features=64*2, k=4, s=2, stddev=0.02, padding=False, relu_factor=0.02, input_dim=self.input_shape[-1], norm_type='Ins'),
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=64*2, out_features=64*4, k=4, s=2, stddev=0.02, padding=False, relu_factor=0.02, input_dim=self.input_shape[-1], norm_type='Ins'),
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=64*4, out_features=64*8, k=4, s=1, stddev=0.02, padding=False, relu_factor=0.02, input_dim=self.input_shape[-1], norm_type='Ins'),
          nn.ZeroPad2d(2),
          General_Conv2D(in_features=64*8, out_features=1, k=4, s=1, stddev=0.02, padding=False, input_dim=self.input_shape[-1], norm_type=None, do_relu=False),
        ])

    def forward(self, x):
      for i in range(len(self.block)):
          x=self.block[i](x)

      return torch.unsqueeze(x[...,0], dim=3), torch.unsqueeze(x[...,1], dim=3)

class Segmenter(nn.Module):
    def __init__(self, keep_rate):
      super(Segmenter, self).__init__()

      self.l1 = General_Conv2D(in_features=64*8, out_features=5, k=1, s=1, stddev=0.01, padding=True, relu_factor=0, do_relu=False, keep_rate=keep_rate, norm_type=None)
      
    def forward(self, x):
      x=self.l1(x)
      x=torch.nn.functional.interpolate(x, size=(65,65), mode='bilinear', align_corners=None)
      return x

#####################################################
# MODEL
#####################################################