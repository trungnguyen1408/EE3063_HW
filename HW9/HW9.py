import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

from PIL import Image
import os
import sys
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Transform data
data_transforms = {
    'train': transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.ColorJitter(saturation=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
    
    'val': transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
}

# Load data
path = './image'
batch_size = 8
num_classes = 10


data = {x: datasets.ImageFolder(os.path.join(path, x), data_transforms[x]) 
              for x in ['train', 'val']}

# Load data
train_data = DataLoader(data['train'], batch_size, shuffle=True)
val_data = DataLoader(data['val'], batch_size, shuffle=True)

train_size = len(data['train'])
val_size = len(data['val'])

class_names = data['train'].classes

# Load model
alexnet = models.alexnet(pretrained=True)
model = alexnet

for para in model.parameters():
  para.requires_grad = False

model.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
model.classifier.add_module("7", nn.LogSoftmax(dim = 1))
model.to(device)

# Trainning
def train_model(model, criterion, optimizer, num_epochs=5):
    since = time.time()
    history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # Trainning
        model.train()
        for train_x, train_y in train_data:
          train_x = train_x.to(device)
          train_y = train_y.to(device)
          
          output_train = model(train_x)
          loss_train = criterion(output_train, train_y)

          # zero the parameter gradients
          optimizer.zero_grad()
          loss_train.backward()
          
          # gradient descent or adam step
          optimizer.step()

          train_loss += loss_train.item() * train_x.size(0)

          _, train_pred = torch.max(output_train.data, 1)
          correct_train = train_pred.eq(train_y.data.view_as(train_pred))
          acc_train = torch.mean(correct_train.type(torch.FloatTensor))
          train_acc += acc_train.item() * train_x.size(0)

        # Validation
        with torch.no_grad():
          model.eval()

          for val_x, val_y in val_data:
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            
            output_val = model(val_x)
            loss_val = criterion(output_val, val_y)

            val_loss += loss_val.item() * val_x.size(0)

            _, val_pred = torch.max(output_val.data, 1)
            correct_val = val_pred.eq(val_y.data.view_as(val_pred))
            acc_val = torch.mean(correct_val.type(torch.FloatTensor))
            val_acc += acc_val.item() * val_x.size(0)


        avg_train_loss = train_loss / train_size
        avg_train_acc = train_acc / train_size
        avg_val_loss = val_loss / val_size
        avg_val_acc = val_acc / val_size   

        history.append([avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc])    

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', avg_train_loss, avg_train_acc))
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', avg_val_loss, avg_val_acc))

        # deep copy the model
        if avg_val_acc >= best_acc:
          best_acc = avg_val_acc
          best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

criterion = nn.NLLLoss()
# Observe that all parameters are being optimized
optimizer = optim.Adam(alexnet.parameters(), lr=1e-3)

model, history = train_model(model, criterion, optimizer, num_epochs=15)

# Save model
# torch.save(model, "./model_" + str(1) + ".pt")

# Evaluation

# Load image test
path = './test/'
file_name = os.listdir(path)
images = [Image.open(path + name) for name in file_name]

# Get labels
labels = [name.split(name[-6])[0] for name in file_name]

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

predictions = model(transform_imgs.cuda())
output = torch.nn.functional.softmax(predictions, dim=-1)

# Top-3 error rate
top3_prob, top3_id = torch.topk(output, 3)
true = 0

for i in range(top3_id.shape[0]):
  if categories[top3_id[i][0]].startswith(labels[i]):
    true += 1
  if categories[top3_id[i][1]].startswith(labels[i]):
    true += 1
  if categories[top3_id[i][2]].startswith(labels[i]):
    true += 1

print('Top-3 error rate: %.2f' %(100*(1-true/top3_id.shape[0])))

# Top-1 error rate
top1_prob, top1_id = torch.topk(output, 1)
true = 0

for i in range(top1_id.shape[0]):
  if categories[top1_id[i][0]].startswith(labels[i]):
    true += 1

print('Top-1 error rate: %.2f' %(100*(1-true/top1_id.shape[0])))

