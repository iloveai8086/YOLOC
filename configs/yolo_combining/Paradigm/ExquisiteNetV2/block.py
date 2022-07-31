from torch.nn import functional as F

#one side pad num
def pad_num(k_s):
    pad_per_side = int((k_s-1)*0.5)
    return pad_per_side

class SE_LN(nn.Module):
    def __init__(self, cin):
        super(SE_LN, self).__init__()  
        self.gavg = nn.AdaptiveAvgPool2d(1)
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1,x.size()[1])
        x = self.ln(x)
        x = self.act(x)   
        x = x.view(-1,x.size()[1],1,1)
        return x*y

class DFSEBV2(nn.Module):
    def __init__(self, cin, dw_s, is_LN):
        super(DFSEBV2, self).__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.SiLU()
        self.dw1 = nn.Conv2d(cin,cin,dw_s,1,pad_num(dw_s),groups=cin)
        if is_LN:
            self.seln = SE_LN(cin)
        else:
            self.seln = SE(cin,3)
            
        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act2 = nn.Hardswish()
        self.dw2 = nn.Conv2d(cin,cin,dw_s,1,pad_num(dw_s),groups=cin)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.seln(x)
        x = x + y
        
        x = self.pw2(x)       
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dw2(x)
        x = x + y
        #del y
        return x

class FCT(nn.Module):
    def __init__(self, cin, cout):
        super(FCT, self).__init__()
        self.dw = nn.Conv2d(cin,cin,4,2,1,groups=cin,bias=False)
        self.minpool = MinPool2d()
        self.maxpool = nn.MaxPool2d(2,ceil_mode=True)
        self.pw = nn.Conv2d(3*cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        z = self.dw(x)
        y = self.minpool(x)
        x = self.maxpool(x)
        x = torch.cat((x,y,z), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x

class EVE(nn.Module):
    def __init__(self, cin, cout):
        super(EVE, self).__init__()
        self.minpool = MinPool2d()
        self.maxpool = nn.MaxPool2d(2,ceil_mode=True)
        self.pw = nn.Conv2d(2*cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        y = self.minpool(x)
        x = self.maxpool(x)
        x = torch.cat((x,y), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x

class ME(nn.Module):
    def __init__(self, cin, cout):
        super(ME, self).__init__()
        self.maxpool = nn.MaxPool2d(2,ceil_mode=True)
        self.pw = nn.Conv2d(cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.pw(x)
        x = self.bn(x)
        return x

class MinPool2d(nn.Module):
    def forward(self, x):
        x = -F.max_pool2d(-x,2,ceil_mode=True)
        return x

class DW(nn.Module):
    def __init__(self, cin, k_s):
        super().__init__()
        self.dw = nn.Conv2d(cin,cin,k_s,1,pad_num(k_s),groups=cin)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.dw(x)
        x = self.act(x)
        return x
