import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Load image
# path = './images/'
# file_name = os.listdir(path)
# images = [Image.open(path + name) for name in file_name]
# print(len(images))

# # Padding 0
# def expand2square(pil_img, background_color):
#     width, height = pil_img.size
#     if width == height:
#         return pil_img
#     elif width > height:
#         result = Image.new(pil_img.mode, (width, width), background_color)
#         result.paste(pil_img, (0, (width - height) // 2))
#         return result
#     else:
#         result = Image.new(pil_img.mode, (height, height), background_color)
#         result.paste(pil_img, ((height - width) // 2, 0))
#         return result

# for i, name in enumerate(file_name):
#   im_new = expand2square(images[i], (0, 0, 0))
#   im_new.save('./image/' + name, quality=95)


# Load image
path = './image/'
file_name = os.listdir(path)
images = [Image.open(path + name) for name in file_name]
print(len(images))

# Get labels
labels = [name.split(name[-7])[0] for name in file_name]
print(labels)

# Prepare data
preprocess = transforms.Compose([
    transforms.Resize(224),
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


model = alexnet
model.eval()

predictions = model(transform_imgs)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
output = torch.nn.functional.softmax(predictions, dim=-1)

# Read the categories
with open("./imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Accuracy
top1_prob, top1_id = torch.topk(output, 1)
true = 0

for i in range(top1_id.shape[0]):
  if categories[top1_id[i][0]].startswith(labels[i]):
    true += 1

print('Accuracy: %.2f' %(100*(true/top1_id.shape[0])))