import os
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torch.nn as nn

import scipy.io as scio
import torchvision.transforms as transforms
import torchvision.models as models

from torch.autograd import Variable

import se_inception

classes = ['personalLess30','personalLess45','personalLess60','personalLarger60','carryingBackpack',
                'carryingOther','lowerBodyCasual','upperBodyCasual','lowerBodyFormal','upperBodyFormal',
                'accessoryHat','upperBodyJacket','lowerBodyJeans','footwearLeatherShoes','upperBodyLogo',
                'hairLong','personalMale','carryingMessengerBag','accessoryMuffler','accessoryNothing',
                'carryingNothing','upperBodyPlaid','carryingPlasticBags','footwearSandals','footwearShoes',
                'lowerBodyShorts','upperBodyShortSleeve','lowerBodyShortSkirt','footwearSneakers','upperBodyThinStripes', #longskirt,thickstripe
                'accessorySunglasses','lowerBodyTrousers','upperBodyTshirt','upperBodyOther','upperBodyVNeck']
class_len = 35


def default_loader(path):
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






mytransform = transforms.Compose([
    transforms.Resize((299,299)),
    #transforms.Resize((299,299)),       #FIXME:resize
    transforms.ToTensor(),            # mmb,
    ]
)

# torch.utils.data.DataLoader
set = myImageFloder(root = "../deepMAR/data/PETA dataset",label = "../deepMAR/testdata.txt", transform = mytransform )
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size= 1, shuffle= True, num_workers= 2)


print len(set)






""" path = "./test/checkpoint_epoch_0"                     #FIXME:PATH
net = Incep.Inception3(num_classes=35)
net.load_state_dict(torch.load(path)) """
historyPath = "./se_inception/checkpoint_epoch_20" 
net = se_inception.se_inception_v3(num_classes=35)
net.load_state_dict(torch.load(historyPath))
net.eval()
net.cuda()

dataiter = iter(imgLoader)

count = 0

TP = [0.0] * 35
P  = [0.0] * 35
TN = [0.0] * 35
N  = [0.0] * 35

Acc = 0.0
Prec = 0.0
Rec = 0.0
while count < 7600:
    images,labels = dataiter.next()
    with torch.no_grad():
        inputs, labels = Variable(images).cuda(), Variable(labels).cuda()
        #a = time.time()
        outputs = net(inputs)
        #b = time.time()
        #print(b-a)    
        Yandf = 0.1
    Yorf = 0.1
    Y = 0.1
    f = 0.1

    i = 0
    for item in outputs[0]:
            if item.item() > 0 :
                f = f + 1
                Yorf = Yorf + 1
                if labels[0][i].item() == 1:
                    TP[i] = TP[i] + 1
                    P[i] = P[i] + 1
                    Y = Y + 1
                    Yandf = Yandf + 1
                else : 
                    N[i] = N[i]  + 1
            else :
                if labels[0][i].item() == 0 :
                    TN[i] = TN[i] + 1
                    N[i] = N[i] + 1
                else:
                    P[i] = P[i] + 1
                    Yorf = Yorf + 1
                    Y = Y + 1
            i = i + 1 
    Acc = Acc +Yandf/Yorf
    Prec = Prec + Yandf/f
    Rec = Rec + Yandf/Y
    if count % 1000 == 0:
        print(count)      
    count = count + 1

Accuracy = 0
print(TP)
print(TN)
print(P)
print(N)
for l in range(35):
    print( "%s : %f" %(classes[l],(TP[l]/P[l] + TN[l]/N[l])/2))
    Accuracy =  TP[l]/P[l] + TN[l]/N[l] + Accuracy
meanAccuracy = Accuracy / 70

print("path: %s mA: %f"%(path,meanAccuracy))

Acc = Acc/7600
Prec = Prec/7600
Rec = Rec/7600
F1 = 2 * Prec * Rec / (Prec + Rec)

print("ACC: %f"%(Acc))
print("Prec: %f"%(Prec))
print("Rec: %f"%(Rec))
print("F1: %f"%(F1))
