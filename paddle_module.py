class MR_FDUM(nn.Module):
    def __init__(self):
        super(MR_FDUM, self).__init__()
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(1)
        self.maxpool = paddle.nn.AdaptiveMaxPool2D(1)
        self.avg_bn = nn.BatchNorm2D(2048)
        self.max_bn = nn.BatchNorm2D(2048)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_avg = self.avgpool(x)
        x_avg = self.avg_bn(x_avg)
        x_max = self.maxpool(x)
        x_max = self.max_bn(x_max)
        super_x = self.relu(x_avg + x_max)
        return  super_x

class spatial_gcn(nn.Module):
    def __init__(self,  in_features, out_features, dropout = False, bias=False,normalize=False):
        super(spatial_gcn, self).__init__()
        self.adj = Parameter(paddle.to_tensor (norm(genA(out_features))).float())
        self.normalize = normalize
        self.dropout = dropout
        self.bn = nn. BatchNorm2D (in_features, eps=1e-04)
        self.in_features = in_features
        self.out_features = out_features
        self.relu = nn.LeakyReLU (0.2)
        self.weight = Parameter(paddle.to_tensor ( in_features, out_features**2))
        if bias:
            self.bias = Parameter(paddle.to_tensor (1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x):
        B = x.size(0)
        X = x.view(B, self.in_features, -1) 
        AA = self.adj.expand(B,  self.out_features**2, self.out_features**2)
        support = paddle.bmm (X, AA )
        spatial_gcn = paddle.multiply (support, self.weight) 
        spatial_ori = spatial_gcn.view(B, self.in_features, *x.size()[2:])
        out =  self.relu(x + self.bn((spatial_ori)))
        if self.bias is not None:
            return out + self.bias
        else:
            return out
            
class FourDirPooling(nn.Module):
    def __init__(self, pooling_type):
        super(FourDirPooling, self).__init__()
        self.hap = paddle.nn.AvgPool2D((20, 32), stride=4)
        self.vap = paddle.nn.AvgPool2D((32, 20), stride=4)
        self.pooling_type = pooling_type
        self.diagonal = Diagonal()
    def forward(self, x):
        batch_size, n_chanel, h, w = x.size()

        if self.pooling_type == 'hap':
            hap_x = self.hap(x)
            return hap_x

        if self.pooling_type == 'vap':
            vap_x = self.vap(x)
            return vap_x

        #anti_diagnoal
        if self.pooling_type == 'aap':
            anti_x = paddle.flip(x, dims=[3])
            aap_x = paddle.zeros((batch_size, n_chanel, 4, 1)).cuda()
            anti_diag = paddle.zeros((batch_size, n_chanel, 63)).cuda()
            j = 0
            for i in range(31, -1, -1):
                anti_diag[:, :, j] = self.diagonal(anti_x[:, :, 0:32 - i, i:], dim1=2, dim2=3).sum(dim=2)
                j = j + 1
            for i in range(1, 32):
                anti_diag[:, :, j] = self.diagonal(anti_x[:, :, i:, :32 - i], dim1=2, dim2=3).sum(dim=2)
                j = j + 1

            aap_x[:, :, 0,0] = anti_diag[:, :, 0:16].sum(dim=2).div(136.0)
            aap_x[:, :, 1,0] = anti_diag[:, :, 16:32].sum(dim=2).div(392.0)
            aap_x[:, :, 2,0] = anti_diag[:, :, 31:47].sum(dim=2).div(392.0)
            aap_x[:, :, 3,0] = anti_diag[:, :, 47:63].sum(dim=2).div(136.0)
            return aap_x

        #diagnoal
        if self.pooling_type == 'dap':
            dap_x = paddle.zeros((batch_size, n_chanel, 4, 1)).cuda()
            diag = paddle.zeros((batch_size, n_chanel, 63)).cuda()
           
            j = 0
            for i in range(31, -1, -1):
                diag[:, :, j] = self.diagonal(x[:, :, 0:32 - i, i:], dim1=2, dim2=3).sum(dim=2)
                
                j = j + 1

            for i in range(1, 32):
                diag[:, :, j] = self.diagonal(x[:, :, i:, :32 - i], dim1=2, dim2=3).sum(dim=2)
                
                j = j + 1

            dap_x[:, :, 0,0] = diag[:, :, 0:16].sum(dim=2).div(136.0)
            dap_x[:, :, 1,0] = diag[:, :, 16:32].sum(dim=2).div(392.0)
            dap_x[:, :, 2,0] = diag[:, :, 31:47].sum(dim=2).div(392.0)
            dap_x[:, :, 3,0] = diag[:, :, 47:63].sum(dim=2).div(136.0)

            return dap_x
