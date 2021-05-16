import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import models




class m_SCNN(nn.Module):
    def __init__(self,out_classes,img_channel=3,msg_pass_kerSize=9,pretrained=True):
        super(m_SCNN, self).__init__()
        self.pretrained = pretrained
        self.classes = out_classes
        self.img_chan = img_channel
        self.net_init(self.classes,self.img_chan,msg_pass_kerSize)
        self.sigm = nn.Sigmoid()
        if not pretrained:
            self.weight_init()
    def net_init(self,classes,im_channel,msg_pass_kerSize):
        # inp_w,inp_h = inp_size
        # self.fc_inp_feature = 5 * int(inp_w/16) * int(inp_h/16)
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features
        #if img 1 channel remove 3 channel add 1 chann
        if im_channel == 1:
            new_chan = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            self.backbone._modules["0"] = new_chan
        # print(self.backbone)
        # exit()


        #------ Process backbone ----
        for i in [34,37,40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels,conv.out_channels,conv.kernel_size,
                stride=conv.stride,padding=tuple(p * 2 for p in conv.padding),
                dilation=2,bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop("33")
        self.backbone._modules.pop("43")
        #
        # print(self.backbone)
        # exit()

        #------ SCNN part
        self.layer1 = nn.Sequential(
            nn.Conv2d(512,1024,3,padding=4,dilation=4,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024,128,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        #----- add message passing ---
        self.msg_passing = nn.ModuleList()
        self.msg_passing.add_module("up_down",nn.Conv2d(128,128,(1,msg_pass_kerSize),
                                                        padding=(0,msg_pass_kerSize//2),
                                                        bias=False))
        self.msg_passing.add_module("down_up",nn.Conv2d(128,128,(1,msg_pass_kerSize),
                                                        padding=(0,msg_pass_kerSize//2),
                                                        bias=False))
        self.msg_passing.add_module("left_right",nn.Conv2d(128,128,(msg_pass_kerSize,1),
                                                           padding=(msg_pass_kerSize//2,0),
                                                           bias=False))

        self.msg_passing.add_module("right_left",nn.Conv2d(128,128,(msg_pass_kerSize,1),
                                                           padding=(msg_pass_kerSize//2,0),
                                                           bias=False))

        self.layer2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(128,classes,1),
        )

    def msg_passing_once(self,x,conv,vertical=True,reverse=False):
        nB,C,H,W = x.shape
        if vertical:
            slices = [x[:,:,i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:,:,:,i:(i+1)] for i in range(W)]
            dim = 3

        if reverse:
            slices = slices[::-1]
        out = [slices[0]]
        for i in range(1,len(slices)):
            out.append(slices[i] + f.relu(conv(out[i - 1])))

        if reverse:
            out = out[::-1]
        return torch.cat(out,dim=dim)

    def msg_passing_forward(self,x):
        vertical = [True,True,False,False]
        revers = [False,True,False,True]

        for msg_conv,v,r in zip(self.msg_passing,vertical,revers):
            x = self.msg_passing_once(x,msg_conv,v,revers)
        return x


    def forward(self,img):
        x = self.backbone(img)
        x = self.layer1(x)
        x = self.msg_passing_forward(x)
        x = self.layer2(x)

        seg_pred = f.interpolate(x,scale_factor=8,mode="bilinear",align_corners=True)

        x = self.sigm(seg_pred)
        return x

    #this needs when testing model
    def predict(self):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward()
        return x

# if __name__ == '__main__':
#     net = m_SCNN(out_classes=5,img_channel=3)
#     x = torch.rand(2,3,512,512)
#     print(net(x).shape)




