import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load image
path = './image/'
file_name = os.listdir(path)
images = [Image.open(path + name) for name in file_name]
print(len(images))

#Get labels
labels = [name.split(name[-6])[0] for name in file_name]
print(labels)


# Prepare data

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_imgs = torch.empty(0, 3, 224, 224)
for img in images:
  tf_img = preprocess(img)
  tensor_img = torch.unsqueeze(tf_img, 0)
  transform_imgs = torch.cat((transform_imgs, tensor_img), dim = 0)

print(transform_imgs.shape)

# Load model

alexnet = models.alexnet(pretrained=True)

# Evaluation
model = alexnet
model.eval()

predictions = model(transform_imgs)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
output = torch.nn.functional.softmax(predictions, dim=-1)

# Read the categories
with open("./imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Show top categories per image
top5_prob, top5_id = torch.topk(output, 5)
true = 0
error = []

for i in range(top5_id.shape[0]):
  if categories[top5_id[i][0]].startswith(labels[i]):
    true += 1
  elif categories[top5_id[i][1]].startswith(labels[i]):
    true += 1
  elif categories[top5_id[i][2]].startswith(labels[i]):
    true += 1
  elif categories[top5_id[i][3]].startswith(labels[i]):
    true += 1
  elif categories[top5_id[i][4]].startswith(labels[i]):
    true += 1
  else:
    error.append(i)

print('Top 5 error rate: %.2f' %(100*(1-true/top5_id.shape[0])))

# Show top categories per image
top1_prob, top1_id = torch.topk(output, 1)
true = 0

for i in range(top1_id.shape[0]):
  if categories[top1_id[i][0]].startswith(labels[i]):
    true += 1

print('Top 1 error rate: %.2f' %(100*(1-true/top1_id.shape[0])))