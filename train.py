import os
import torch
import torch.utils.data as data
from PIL import Image

import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torchvision.models as models


import se
import se2
import se_attribute
import se_info
import torchvision.transforms as transforms

import  random
random.seed(1)
from visdom import Visdom
import numpy as np

""" import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-resnet',default = False)
args = parser.parse_args() """


viz = Visdom()
win = viz.line(
    X=np.array([0]),
    Y=np.array([0]),
    name="1"
)
win2 = viz.line(
    X=np.array([0]),
    Y=np.array([0]),
    name="2"
)

class_names = ['personalLess30','personalLess45','personalLess60','personalLarger60','carryingBackpack',
                'carryingOther','lowerBodyCasual','upperBodyCasual','lowerBodyFormal','upperBodyFormal',
                'accessoryHat','upperBodyJacket','lowerBodyJeans','footwearLeatherShoes','upperBodyLogo',
                'hairLong','personalMale','carryingMessengerBag','accessoryMuffler','accessoryNothing',
                'carryingNothing','upperBodyPlaid','carryingPlasticBags','footwearSandals','footwearShoes',
                'lowerBodyShorts','upperBodyShortSleeve','lowerBodyShortSkirt','footwearSneakers','upperBodyThinStripes', #longskirt,thickstripe
                'accessorySunglasses','lowerBodyTrousers','upperBodyTshirt','upperBodyOther','upperBodyVNeck']
class_len = 35


def default_loader(path):
    if random.random()<0.6:
        return Image.open(path).convert('RGB')
    else:
        x0 = random.randint(0,29)
        y0 = random.randint(0,29)
        return Image.open(path).resize((256,256),Image.BILINEAR).crop((x0,y0,x0+227,y0+227)).convert('RGB')

def default_loader2(path):

    return Image.open(path).convert('RGB')

class myImageFloder(data.Dataset):
    def __init__(self, root, label,transform = None, target_transform = None, loader = default_loader):
        
        
        imgs=[]
        
        
        file = open(label)
        for line in file.readlines():
            cls = line.split()
            pos = cls.pop(0)
            att_vector = []
            for att in cls:
                if att == '0':
                    att_vector.append(0)
                else:
                    att_vector.append(1)
            if os.path.isfile(os.path.join(root,pos)):
                #imgs.append((name[0][0],[x*2-1 for x in testlabel[count]]))   (-1,1)
                imgs.append((pos,att_vector))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root,fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)
    

def imshow(imgs):
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.title("bat")
    plt.show()

def checkpoint(epoch):
    if not os.path.exists("se_i"):
        os.mkdir("se_i")
    path = "./se_i/checkpoint_epoch_{}".format(epoch)

    torch.save(net.state_dict(),path)

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)



def evaluate(imgloader,net):
    dataiter = iter(imgloader)
    net.eval()
    count = 0

    TP = [0.0] * 35
    P  = [0.0] * 35
    TN = [0.0] * 35
    N  = [0.0] * 35

    while count < 1900:
        images,labels = dataiter.next()
        with torch.no_grad():
            inputs, labels = Variable(images).cuda(), Variable(labels).cuda()
            #a = time.time()
            outputs = net(inputs)
            #b = time.time()
            #print(b-a)    


        i = 0
        for item in outputs[0]:
                if item.item() > 0 :

                    if labels[0][i].item() == 1:
                        TP[i] = TP[i] + 1
                        P[i] = P[i] + 1

                    else : 
                        N[i] = N[i]  + 1
                else :
                    if labels[0][i].item() == 0 :
                        TN[i] = TN[i] + 1
                        N[i] = N[i] + 1
                    else:
                        P[i] = P[i] + 1

                i = i + 1 

        count = count + 1

    Accuracy = 0
    for l in range(35):
        Accuracy =  TP[l]/P[l] + TN[l]/N[l] + Accuracy
    meanAccuracy = Accuracy / 70
    net.train()
    return meanAccuracy



mytransform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.Resize((328,328)),
    #transforms.RandomCrop((299,299)),
    transforms.Resize((299,299)),       #TODO:maybe need to change1
    transforms.ToTensor(),            # mmb,
    ]
)

mytransform2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.Resize(328),
    #transforms.RandomCrop(299),
    transforms.Resize((299,299)),       #TODO:maybe need to change1
    transforms.ToTensor(),            # mmb,
    ]
)

# torch.utils.data.DataLoader
set = myImageFloder(root = "../data/PETA dataset",label = "../deepMAR/traindata.txt",
                     transform = mytransform, loader = default_loader2 )
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size= 16, shuffle= True, num_workers= 2)

set2 = myImageFloder(root = "../data/PETA dataset",label = "../deepMAR/valdata.txt", 
                transform = mytransform2 ,loader = default_loader2)
testLoader = torch.utils.data.DataLoader(
         set2, 
         batch_size= 1, shuffle= True, num_workers= 2)
print len(set2)

print len(set)




'''net = Incep.Inception3(num_classes=35)

net.apply(weight_init)


net_dict = net.state_dict()
path = "./checkpoint_epoch_0" 
pretrained_dict = torch.load(path)
pretrained_dict = {k : v for k,v in pretrained_dict.items() if k in net_dict and pretrained_dict[k].size() == net_dict[k].size()}
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)  '''


""" net = se_attribute.senet(num_classes=35)
#net.apply(weight_init)
path = "./se_a/checkpoint_epoch_20"
net.load_state_dict(torch.load(path))  """


net = se_info.senet(num_classes=35)
net.apply(weight_init)
net_dict = net.state_dict()
path = "mnet_0.81" 
pretrained_dict = torch.load(path)
pretrained_dict = {k : v for k,v in pretrained_dict.items() if k in net_dict and pretrained_dict[k].size() == net_dict[k].size()}
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)


for param in net.parameters():
    param.requires_grad = False


for param in net.fc.parameters():
    param.requires_grad = True
for param in net.info.parameters():
    param.requires_grad = True
for param in net.lstm1.parameters():
    param.requires_grad = True
for param in net.lstm2.parameters():
    param.requires_grad = True
for param in net.lstm3.parameters():
    param.requires_grad = True
for param in net.lstm4.parameters():
    param.requires_grad = True
net.train()
net.cuda()


weight = torch.Tensor([1.6525437864263877, 1.9595930852167125, 2.4541720986628794, 2.5540598994497885, 2.2299964644001817, 
2.2370496893077894, 1.1530620461497678, 1.1623235390961753, 2.3626632385568676, 2.3733816376087318, 
2.4583089238092017, 2.5270500431133978, 2.0063472685226964, 2.012057629534014, 2.6166496419453167, 
2.1436850107543335, 1.5787471334106629, 2.027791699997802, 2.497958215072316, 1.2821345623085803, 
2.059196358542684, 2.6518649783719157, 2.5185521947476883, 2.6655783544594476, 1.8966805193783818, 
2.630734692383632, 2.360674464904516, 2.594980357872065, 2.1932148184165525, 2.6748538326153977,
 2.6387776259922835, 1.6199917435084439, 2.5008522629204433, 1.72706088381075, 2.6867056817110972])
criterion = nn.BCEWithLogitsLoss(weight = weight)
criterion.cuda()

#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,net.parameters()), lr=0.001, momentum = 0.9)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

running_loss = 0.0
for epoch in range(1000):
    for i, data in enumerate(imgLoader, 0):
            # get the inputs
            inputs, labels = data
            
            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            #inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            


            # forward + backward + optimize
            outputs = net(inputs)
            
            loss = criterion(outputs, labels) 
            #print(loss)
            loss.backward()        
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 1000 == 0: # print every 2000 mini-batches
                print('[ %d %5d] loss: %.6f' % ( epoch,i+1, running_loss / 100))
                viz.line(
                    X=np.array([epoch+i/1187.5]),
                    Y=np.array([running_loss]),
                    win=win,
                    update='append',  #new version need
                    name="1"
                )
                running_loss = 0.0

    ma = evaluate(testLoader,net)
    print(ma)
    viz.line(
                    X=np.array([epoch]),
                    Y=np.array([ma]),
                    win=win2,
                    update='append',
                    name="2"
                )    
    if epoch % 10 == 0:
        checkpoint(epoch%50)
"""         for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9
 """













