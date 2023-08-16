# codes
## deep learning
### a very simple code for training the resnet18 from scratch

#### load the data
```py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def load_data(data_dir = "../hw2_dataset_2022/",input_size = 224,batch_size = 36):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'train_augmented': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomRotation(degrees=55),
            # transforms.RandomRotation([90, 180]),
            transforms.RandomHorizontalFlip(),
            # RandomErasing(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    ## The default dir is for the first task of large-scale deep learning
    ## For other tasks, you may need to modify the data dir or even rewrite some part of 'data.py'
    ## ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名

    image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, '1-Large-Scale', 'train'), data_transforms['train'])
    image_dataset_valid = datasets.ImageFolder(os.path.join(data_dir,'test'), data_transforms['test'])

    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader



```

#### define the models
```py
from torchvision import models
import torch.nn as nn


def model_A(num_classes):
    model_resnet = models.resnet18(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet


```

#### train and test
```py
import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=20):

    def train(model, train_loader,optimizer,criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            
            

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        # 更新学习率
        # scheduler.step()
        # 获取当前的学习率
        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item(), current_lr
    
    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predictions = torch.max(outputs, 1)
                total_loss += loss.item() * inputs.size(0)
                total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    train_loss_epoch = []
    train_acc_epoch = []
    valid_loss_epoch = []
    valid_acc_epoch = []
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc, current_lr = train(model, train_loader,optimizer,criterion)
        train_loss_epoch.append(train_loss)
        train_acc_epoch.append(train_acc)
        print("training: {:.4f}, {:.4f}, lr:{}".format(train_loss, train_acc, current_lr))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        valid_loss_epoch.append(valid_loss)
        valid_acc_epoch.append(valid_acc)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model.pt')

        

    
    plt.plot(train_loss_epoch)
    plt.ylabel('Train Loss')
    plt.xlabel('Epoch')
    plt.savefig('B-Train Loss.jpg')
    plt.close()

    plt.plot(train_acc_epoch)
    plt.ylabel('Train Acc')
    plt.xlabel('Epoch')
    plt.savefig('B-Train Acc.jpg')
    plt.close()

    plt.plot(valid_loss_epoch)
    plt.ylabel('Test Loss')
    plt.xlabel('Epoch')
    plt.savefig('B-Test Loss.jpg')
    plt.close()

    plt.plot(valid_acc_epoch)
    plt.ylabel('Test Acc')
    plt.xlabel('Epoch')
    plt.savefig('B-Test Acc.jpg')
    plt.close()

    

def test_model(model, valid_loader):
    model.train(False)
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            total_correct += torch.sum(predictions == labels.data)
        
    epoch_acc = total_correct.double() / len(valid_loader.dataset)
    print("test_acc: {:.4f}".format(epoch_acc.item()))

    
def plot_tsne(model, valid_loader):
    fc_features = []
    def hook(model, input, output):
        fc_features.append(input[0].cpu().detach().numpy())
    model.fc.register_forward_hook(hook)
    model.train(False)

    y = []

    for inputs, labels in valid_loader:
        y.append(labels.numpy())
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

    fc_feature = np.concatenate(fc_features)
    y = np.concatenate(y)

    tsne = TSNE(n_components=2)
    fc_embeded = tsne.fit_transform(fc_feature)
    plt.scatter(x=fc_embeded[:, 0], y=fc_embeded[:, 1], c=y, cmap='rainbow', s=1.5)
    plt.savefig('features.jpg')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## about model
    num_classes = 10

    ## about data
    data_dir = "../../hw2_dataset_2022" ## You need to specify the data_dir first
    input_size = 224
    batch_size = 36

    ## about training
    num_epochs = 100
    lr = 0.001

    # 定义L2正则化参数
    l2_lambda = 0.001

    ## model initialization
    model = models.model_A(num_classes=num_classes)
    #model = models.model_B(num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=input_size, batch_size=batch_size)

    ## optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    


    ## loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)
    model = torch.load('best_model.pt')
    test_model(model, valid_loader)
    # plot_tsne(model, valid_loader)


```