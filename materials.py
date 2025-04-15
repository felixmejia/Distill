'''Material classes for 'run_chexpert.py'''

###################
## Prerequisites ##
###################
import time
import pickle
import random
import csv
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score


from tqdm import tqdm



use_gpu = torch.cuda.is_available()



######################
## Create a Dataset ##
######################
class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, nnClassCount, policy, transform = None):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(data_PATH, 'r') as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[0]
                npline = np.array(line)
                idx = [7, 10, 11, 13, 15]
                label = list(npline[idx])
                for i in range(nnClassCount):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == 'diff':
                                if i == 1 or i == 3 or i == 4:  # Atelectasis, Edema, Pleural Effusion
                                    label[i] = 1                    # U-Ones
                                elif i == 0 or i == 2:          # Cardiomegaly, Consolidation
                                    label[i] = 0                    # U-Zeroes
                            elif policy == 'ones':              # All U-Ones
                                label[i] = 1
                            else:
                                label[i] = 0                    # All U-Zeroes
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                image_names.append('./' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        '''Take the index of item and returns the image and its labels'''
        image_name = self.image_names[index]
        # path_image='NOMBRE ARCHIVO == {0} '
        # print(path_image.format(image_name))
        image_name = image_name.replace("./CheXpert-v1.0-small", "/home/oem/WORKS/DATA/CheXpert-v1.0-small")

        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


############################
## Create CheXpertTrainer ##
############################
class CheXpertTrainer():
    def train(model, dataLoaderTrain, dataLoaderVal, class_names, nnClassCount, trMaxEpoch, PATH, f_or_l, checkpoint, cfg, namefile):
        optimizer = optim.Adam(model.parameters(), lr = cfg.lr, # setting optimizer & scheduler
                               betas = tuple(cfg.betas), eps = cfg.eps, weight_decay = cfg.weight_decay) 
        loss = torch.nn.BCELoss() # setting loss function

        if checkpoint != None and use_gpu: # loading checkpoint
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            model = model.cuda()

        print('<<< Training & Evaluating ({}) >>>'.format(f_or_l))
        # check initial model valid set performance
        lossv1, lossv_each = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)
        print("Initial valid loss (overall): {:.3f}".format(lossv1))
        for i in range(5):
            print("Initial valid loss {}: {:.3f}".format(class_names[i], lossv_each[i]))
        print('')

        # Train the network
        lossMIN, lossMIN_each = 100000, [100000]*5
        lossv_traj_epoch = np.empty((nnClassCount, 0)).tolist()
        model_num_each = [0]*5
        train_start, train_end = [], []
        
        for epochID in range(0, trMaxEpoch):
            train_start.append(time.time()) # training starts
            losst = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, nnClassCount, loss, PATH, f_or_l, epochID, trMaxEpoch, namefile)
            train_end.append(time.time())   # training ends

            lossv, lossv_each = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)
            for i in range(5):
                lossv_traj_epoch[i].append(lossv_each[i])
                if lossv_each[i] < lossMIN_each[i]:
                    lossMIN_each[i] = lossv_each[i]
                    model_num_each[i] = epochID + 1
                    print('Epoch ' + str(epochID + 1) + ' [IMPR] lossv {} = {:.3f}'.format(class_names[i], lossv_each[i]))
                else:
                    print('Epoch ' + str(epochID + 1) + ' [----] lossv {} = {:.3f}'.format(class_names[i], lossv_each[i]))

            if lossv < lossMIN:
                lossMIN = lossv
                model_num = epochID + 1
                print('Epoch ' + str(epochID + 1) + ' [IMPR] loss = {:.3f}'.format(lossv))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] loss = {:.3f}'.format(lossv))

            print("Training loss: {:.3f},".format(losst), "Valid loss: {:.3f}".format(lossv))
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                        'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 
                        '{0}m_{1}-epoch_{2}_{3}.pth.tar'.format(PATH, namefile, epochID + 1, f_or_l))

        print('')
        train_time = np.array(train_end) - np.array(train_start)
        with open("{0}{1}_{2}_lossv_traj_epoch.txt".format(PATH, f_or_l, namefile), "wb") as fp:
            pickle.dump(lossv_traj_epoch, fp)

        return model_num, model_num_each, train_time
       
        
    def epochTrain(model, dataLoaderTrain, optimizer, nnClassCount, loss, PATH, f_or_l, epochID, trMaxEpoch, namefile):
        step=0
        model.train()
        losstrain = 0
        with tqdm(total=len(dataLoaderTrain), desc='Step at start {}; Training {} epoch {}/{}'.format(step, namefile, epochID+1, trMaxEpoch)) as pbar:
    
            for batchID, (varInput, target) in enumerate(dataLoaderTrain):
                optimizer.zero_grad()
                step += 1
                
                varTarget = target.cuda(non_blocking = True)
                varOutput = model(varInput)
                lossvalue = loss(varOutput, varTarget)
                        
                lossvalue.backward()
                optimizer.step()
                pbar.set_postfix(loss = '{:.4f}'.format(lossvalue.item()))
                pbar.update()

                losstrain += lossvalue.item()*varInput.size(0)
                if batchID % 1000 == 999:
                    print('[Batch: %5d] loss: %.3f'%(batchID + 1, losstrain / 1000))

        return losstrain / len(dataLoaderTrain.dataset)
    
    
    def epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss):
        model.eval()
        lossVal = 0
        lossVal_Card, lossVal_Edem, lossVal_Cons, lossVal_Atel, lossVal_PlEf = 0, 0, 0, 0, 0

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoaderVal):
                
                target = target.cuda(non_blocking = True)
                varOutput = model(varInput)

                varOutput_Card = torch.tensor([i[0] for i in varOutput.tolist()])
                target_Card = torch.tensor([i[0] for i in target.tolist()])
                varOutput_Edem = torch.tensor([i[1] for i in varOutput.tolist()])
                target_Edem = torch.tensor([i[1] for i in target.tolist()])
                varOutput_Cons = torch.tensor([i[2] for i in varOutput.tolist()])
                target_Cons = torch.tensor([i[2] for i in target.tolist()])
                varOutput_Atel = torch.tensor([i[3] for i in varOutput.tolist()])
                target_Atel = torch.tensor([i[3] for i in target.tolist()])
                varOutput_PlEf = torch.tensor([i[4] for i in varOutput.tolist()])
                target_PlEf = torch.tensor([i[4] for i in target.tolist()])

                lossvalue = loss(varOutput, target)
                lossVal += lossvalue.item()*varInput.size(0)
                
                lossVal_Card += loss(varOutput_Card, target_Card).item()*varInput.size(0)
                lossVal_Edem += loss(varOutput_Edem, target_Edem).item()*varInput.size(0)
                lossVal_Cons += loss(varOutput_Cons, target_Cons).item()*varInput.size(0)
                lossVal_Atel += loss(varOutput_Atel, target_Atel).item()*varInput.size(0)
                lossVal_PlEf += loss(varOutput_PlEf, target_PlEf).item()*varInput.size(0)
                
            lossv = lossVal / len(dataLoaderVal.dataset)
            lossv_Card = lossVal_Card / len(dataLoaderVal.dataset)
            lossv_Edem = lossVal_Edem / len(dataLoaderVal.dataset)
            lossv_Cons = lossVal_Cons / len(dataLoaderVal.dataset)
            lossv_Atel = lossVal_Atel / len(dataLoaderVal.dataset)
            lossv_PlEf = lossVal_PlEf / len(dataLoaderVal.dataset)
            lossv_each = [lossv_Card, lossv_Edem, lossv_Cons, lossv_Atel, lossv_PlEf]
                                
        return lossv, lossv_each

    
    def computeAUROC(dataGT, dataPRED, nnClassCount):
        # Computes area under ROC curve 
        # dataGT: ground truth data
        # dataPRED: predicted data
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(nnClassCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC
    
    
    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names, f_or_l, nameModel):
        cudnn.benchmark = True
        
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        model.eval()
        outPROB = []
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):

                target = target.cuda()
                outGT = torch.cat((outGT, target), 0).cuda()
                outProb = model(input) # probability
                outProb = outProb.tolist()
                outPROB.append(outProb)

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
                
                out = model(varInput)
                outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        print('<<< Model Test Results: AUROC ({0})  {1} >>>'.format(f_or_l, nameModel))
        print('MEAN', ': {:.4f}'.format(aurocMean))
        
        for i in range (0, len(aurocIndividual)):
            print(class_names[i], ': {:.4f}'.format(aurocIndividual[i]))
        print('')
        return outGT, outPRED, outPROB, aurocMean, aurocIndividual



##################
## Define Model ##
##################
class DenseNet121(nn.Module):
    '''Model modified.
    The architecture of this model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    '''
    def __init__(self, out_size, nnIsTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained = nnIsTrained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class CheXpertMobileNetV3(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CheXpertMobileNetV3, self).__init__()
        
        # Cargar MobileNetV3-Small (más ligero que V2)
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.mobilenet = torchvision.models.mobilenet_v3_small(weights=weights)
        
        # Ajustar la capa clasificadora
        in_features = self.mobilenet.classifier[-1].in_features  # Última capa lineal
        self.mobilenet.classifier[-1] = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()  # Para clasificación multi-etiqueta
        )

    def forward(self, x):
        return self.mobilenet(x)
    


###############################
## Define Ensembles Function ##
###############################
def EnsemAgg(EnsemResult, dataLoader, nnClassCount, class_names):
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    with torch.no_grad():
        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0).cuda()
            
            bs, c, h, w = input.size()
            varInput = input.view(-1, c, h, w)

            # out = model(varInput)
            out = torch.tensor([EnsemResult[i]]).cuda()
            outPRED = torch.cat((outPRED, out), 0)
    aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
    aurocMean = np.array(aurocIndividual).mean()
    print('<<< Ensembles Test Results: AUROC >>>')
    print('MEAN', ': {:.4f}'.format(aurocMean))
    for i in range (0, len(aurocIndividual)):
        print(class_names[i], ': {:.4f}'.format(aurocIndividual[i]))
    print('')

    return outGT, outPRED, aurocMean, aurocIndividual



def train_knowledge_distillation(teacher, student, data_Train, data_Val, nnClassCount, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device, namefile, f_or_l, PATH):
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction='batchmean')


    class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode


    # Train the network
    lossMIN, lossMIN_each = 100000, [100000]*5
    lossv_traj_epoch = np.empty((nnClassCount, 0)).tolist()
    model_num_each = [0]*5
    train_start, train_end = [], []
        
    for epoch in range(epochs):
        running_loss = 0.0
        step=0
        train_start.append(time.time()) # training starts
        with tqdm(total=len(data_Train), desc='Step at start {}; Training KD epoch {}/{}'.format(step, epoch+1, epochs)) as pbar:
            for batchID, (varInput, target) in enumerate(data_Train):
                inputs, labels = varInput.to(device), target.to(device)

                # print(inputs.device, labels.device)  # Debería imprimir "cuda:0" si está en GPU
                # Para que no imprima en cada iteración
                step += 1
                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    teacher_logits = teacher(inputs)

                # Forward pass with the student model
                student_logits = student(inputs)

                #Soften the student logits by applying softmax first and log() second
                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                # soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
                # Calculate the distillation loss
                soft_targets_loss = kd_loss(soft_prob, soft_targets) * (T**2)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss = '{:.4f}'.format(loss.item()))
                pbar.update()

                running_loss += loss.item()
        avg_loss = running_loss / len(data_Train)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        train_end.append(time.time())   # training ends
        lossv, lossv_each = CheXpertTrainer.epochVal(student, data_Val, optimizer, epochs, nnClassCount, kd_loss)
        for i in range(5):
            lossv_traj_epoch[i].append(lossv_each[i])
            if lossv_each[i] < lossMIN_each[i]:
                lossMIN_each[i] = lossv_each[i]
                model_num_each[i] = epoch + 1
                print('Epoch ' + str(epoch + 1) + ' [IMPR] lossv {} = {:.3f}'.format(class_names[i], lossv_each[i]))
            else:
                print('Epoch ' + str(epoch + 1) + ' [----] lossv {} = {:.3f}'.format(class_names[i], lossv_each[i]))

        if lossv < lossMIN:
            lossMIN = lossv
            model_num = epoch + 1
            print('Epoch ' + str(epoch + 1) + ' [IMPR] loss = {:.3f}'.format(lossv))
        else:
            print('Epoch ' + str(epoch + 1) + ' [----] loss = {:.3f}'.format(lossv))

        print("Training loss: {:.3f},".format(avg_loss), "Valid loss: {:.3f}".format(lossv))
        torch.save({'epoch': epoch + 1, 'state_dict': student.state_dict(), 
                    'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 
                    '{0}m_{1}-epoch_{2}_{3}.pth.tar'.format(PATH, namefile , epoch + 1, f_or_l))
        
    
    print('')
    train_time = np.array(train_end) - np.array(train_start)
    with open("{0}{1}_{2}_lossv_traj_epoch.txt".format(PATH, namefile, f_or_l), "wb") as fp:
        pickle.dump(lossv_traj_epoch, fp)

    
    

    return student, model_num, model_num_each, train_time


