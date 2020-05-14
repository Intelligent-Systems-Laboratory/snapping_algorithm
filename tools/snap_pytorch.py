import torch
import torchvision.transforms as T
from torchvision import models
import imutils
import numpy as np


def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    (255, 255, 255), (255, 255, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (255, 255, 0), (255, 0, 0),
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(img):
    fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
    trf = T.Compose([T.Resize(850), 
            #T.CenterCrop(224), 
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], 
                    std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to('cuda')
    out = fcn.to('cuda')(inp)['out']

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    return rgb