import cv2
import numpy as np
import torch
from torch import nn
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
device

IMAGE_SIZE = 224
NUM_CLASSES = 3

NUM_EPOCHS = 100
NUM_WORKERS = 4
LEARNING_RATE = 0.001

CONV_KERNEL = 3
CONV_STRIDE = 1
CONV_PADDING = 1
MP_KERNEL = 2
MP_STRIDE = 2
MP_PADDING = 0

VGG16_ARCHITECTURE = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

check_point = r'YOUR PATH TO YOUR MODE_WEIGHT.pt'

class VGG(nn.Module):
  def __init__(self, in_channels=3, num_classes=None):
    super(VGG, self).__init__()
    self.in_channels = in_channels
    self.features = self.create_conv_layers(VGG16_ARCHITECTURE)
    self.avgpool = nn.AdaptiveAvgPool2d((7,7))
    self.classifier = nn.Sequential(
        nn.Linear(512*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 1000),
        nn.Linear(1000,512),
        nn.Linear(512, num_classes),
    )

  def forward(self,x):
    x = self.features(x)
    x = x.reshape(x.shape[0], -1)
    x = self.classifier(x)
    return x

  def create_conv_layers(self, archite):
    layers = []
    in_channels = self.in_channels
    for x in archite:
      if type(x) == int:
        out_channels = x
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(CONV_KERNEL,CONV_KERNEL), stride=(CONV_STRIDE,CONV_STRIDE), padding=(CONV_PADDING,CONV_PADDING)),
                  #  nn.BatchNorm2d(x),
                   nn.ReLU()]
        in_channels = x
      elif x == "M":
        layers += [nn.MaxPool2d(kernel_size=(MP_KERNEL,MP_KERNEL), stride=(MP_STRIDE,MP_STRIDE))]
    return nn.Sequential(*layers)



model = VGG(in_channels=3, num_classes=NUM_CLASSES).to(device=device)


def inferences(img_path):
    classes = ["cow", "horse", "sheep"]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    image = image / 255.

    # equivalent to Normalize()
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    image = torch.from_numpy(image).float().to(device)
    
    checkpoint = torch.load(os.path.join(check_point, "best.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        prob = softmax(output)
        predicted_prob, predicted_class = torch.max(prob, dim=1)
        print(predicted_prob, predicted_class)
        print("The image is about {}".format(classes[predicted_class]))
        score = predicted_prob[0]*100
        cv2.imshow("{} with confident score of {:0.2f}%".format(classes[predicted_class[0]], score), cv2.imread(img_path))
        cv2.waitKey(5000) 
        cv2.destroyAllWindows


if __name__ == '__main__':
    img_path = r'inferences_img/sheep_white.jpeg'
    inferences(img_path)