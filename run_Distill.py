'''Running the Model'''

###################
## Prerequisites ##
###################
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3' # should do this before importing torch modules!
import time
import json
import pickle
import random
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from easydict import EasyDict as edict
from materials import CheXpertDataSet, CheXpertTrainer, DenseNet121, CheXpertMobileNetV3, train_knowledge_distillation, EnsemAgg

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

use_gpu = torch.cuda.is_available()


def Train_Model(model, dataLoaderTrain_frt, dataLoaderTrain_lat, dataLoaderVal_frt, dataLoaderVal_lat, class_names, nnClassCount, trMaxEpoch, PATH, config , nameModel ):
    # Train frontal model
    train_valid_start_frt = time.time()
    '''See 'materials.py' to check the class 'CheXpertTrainer'.'''
    model_num_frt, model_num_frt_each, train_time_frt = CheXpertTrainer.train(model, dataLoaderTrain_frt, dataLoaderVal_frt, class_names,
                                                                              nnClassCount, trMaxEpoch, PATH, 'frt', checkpoint = None, cfg = config, namefile = nameModel)
    train_valid_end_frt = time.time()

    # Train lateral model
    train_valid_start_lat = time.time()
    '''See 'materials.py' to check the class 'CheXpertTrainer'.'''
    model_num_lat, model_num_lat_each, train_time_lat = CheXpertTrainer.train(model, dataLoaderTrain_lat, dataLoaderVal_lat, class_names,
                                                                              nnClassCount, trMaxEpoch, PATH, 'lat', checkpoint = None, cfg = cfg, namefile = nameModel)
    train_valid_end_lat = time.time()
    print('<<< Model Trained >>>')
    print('For frontal model,', 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt), 'is the best overall.')
    for i in range(5):
        print('For frontal {0},'.format(class_names[i]), 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt_each[i]), 'is the best.')
    print('')
    print('For lateral model,', 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat), 'is the best overall.')
    for i in range(5):
        print('For lateral {0},'.format(class_names[i]), 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat_each[i]), 'is the best.')
    print('')

    return model, model_num_frt, model_num_frt_each, model_num_lat_each, train_valid_start_frt, train_valid_end_frt, train_time_frt, train_valid_start_lat, train_valid_end_lat, train_time_lat




def Test_ROC(PATH, model, dataLoaderTest_frt, dataLoaderTest_lat, nnClassCount, class_names, nameFile, pathFileTest_frt, pathFileTest_lat):

    ##############################
    ## Test and Draw ROC Curves ##
    ##############################
    checkpoint_frt = PATH + 'm_{0}-epoch_{1}_frt.pth.tar'.format(nameFile, trMaxEpoch) # Use the last model ('model_num_frt' if use valid set for model decision)
    checkpoint_lat = PATH + 'm_{0}-epoch_{1}_lat.pth.tar'.format(nameFile, trMaxEpoch) # Use the last model ('model_num_lat' if use valid set for model decision)
    '''See 'materials.py' to check the class 'CheXpertTrainer'.'''
    outGT_frt, outPRED_frt, outPROB_frt, aurocMean_frt, aurocIndividual_frt = CheXpertTrainer.test(model, dataLoaderTest_frt, nnClassCount, checkpoint_frt, class_names, 'frt', nameFile)
    outGT_lat, outPRED_lat, outPROB_lat, aurocMean_lat, aurocIndividual_lat = CheXpertTrainer.test(model, dataLoaderTest_lat, nnClassCount, checkpoint_lat, class_names, 'lat', nameFile)


    PATH = args.output_path
    if args.output_path[-1] != '/':
        PATH = PATH + '/'
    else:
        PATH = PATH

    if not os.path.exists(PATH): os.makedirs(PATH)

    # Save the test outPROB_frt
    with open('{0}testPROB_{1}_frt.txt'.format(PATH, nameFile), 'wb') as fp:
        pickle.dump(outPROB_frt, fp)

    # Save the test outPROB_lat
    with open('{0}testPROB_{1}_lat.txt'.format(PATH, nameFile), 'wb') as fp:
        pickle.dump(outPROB_lat, fp)

    test_frt = pd.read_csv(pathFileTest_frt)
    test_lat = pd.read_csv(pathFileTest_lat)

    column_names = ['Path'] + class_names
    df = pd.DataFrame(0, index = np.arange(len(test_frt) + len(test_lat)), columns = column_names)
    test_frt_list = list(test_frt['Path'].copy())
    test_lat_list = list(test_lat['Path'].copy())

    for i in range(len(test_frt_list)):
        df.iloc[i, 0] = test_frt_list[i].split('/')[2] + '/' + test_frt_list[i].split('/')[3]

    for i in range(len(test_lat_list)):
        df.iloc[len(test_frt_list) + i, 0] = test_lat_list[i].split('/')[2] + '/' + test_frt_list[i].split('/')[3]

    for i in range(len(outPROB_frt)):
        for j in range(len(class_names)):
            df.iloc[i, j + 1] = outPROB_frt[i][0][j]
            
    for i in range(len(outPROB_lat)):
        for j in range(len(class_names)):
            df.iloc[len(outPROB_frt) + i, j + 1] = outPROB_lat[i][0][j]

    df_agg = df.groupby('Path').agg({'Card' : 'min',
                                    'Edem' : 'max',
                                    'Cons' : 'min',
                                    'Atel' : 'min',
                                    'PlEf' : 'min'}).reset_index()
    df_agg = df_agg.sort_values('Path')
    results = df_agg.drop(['Path'], axis = 1).values.tolist()

    # Save the test outPROB_all
    outPROB_all = []
    for i in range(len(results)):
        outPROB_all.append([results[i]])

    with open('{0}testPROB_{1}_all.txt'.format(PATH, nameFile), 'wb') as fp:
        pickle.dump(outPROB_all, fp)

    # Draw ROC curves
    EnsemTest = results
    '''See 'materials.py' to check the function 'EnsemAgg'.'''
    outGT, outPRED, aurocMean, aurocIndividual = EnsemAgg(EnsemTest, dataLoaderTest_agg, nnClassCount, class_names)

    fig, ax = plt.subplots(nrows = 1, ncols = nnClassCount)
    ax = ax.flatten()
    fig.set_size_inches((nnClassCount * 10, 10))
    for i in range(nnClassCount):
        fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:, i], outPRED.cpu()[:, i])
        roc_auc = metrics.auc(fpr, tpr)
        
        ax[i].plot(fpr, tpr, label = 'AUC = %0.2f' % (roc_auc))
        ax[i].set_title('ROC for: ' + class_names[i])
        ax[i].legend(loc = 'lower right')
        ax[i].plot([0, 1], [0, 1],'r--')
        ax[i].set_xlim([0, 1])
        ax[i].set_ylim([0, 1])
        ax[i].set_ylabel('True Positive Rate')
        ax[i].set_xlabel('False Positive Rate')

    plt.savefig('{0}{1}_ROC_{2}.png'.format(PATH, nameFile, nnClassCount), dpi = 100)
    plt.close()

    return outGT, outPRED, aurocMean, aurocIndividual


######################
## Arguments to Set ##
######################
parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfg_path', metavar = 'CFG_PATH', type = str, help = 'Path to the config file in yaml format.')
parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/')
parser.add_argument('--random_seed', '-s', type = int, help = 'Random seed for reproduction.')
args = parser.parse_args()
with open(args.cfg_path) as f:
    cfg = edict(json.load(f))

# Example running commands ('nohup' command for running background on server)
'''
python3 run_chexpert.py configuration.json 
python3 run_chexpert.py configuration.json -o results/ -s 2021
nohup python3 run_chexpert.py configuration.json -o ensembles/experiment_00/ -s 0 > ensemble/printed_00.txt &
'''

# Control randomness for reproduction
if args.random_seed != None:
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



#######################
## Pre-define Values ##
#######################
'''Should have run 'run_preprocessing.py' before this part!'''
# Paths to the files with training, validation, and test sets.
if cfg.image_type == 'small':
    img_type = '-small'
else:
    img_type = ''



path = '/home/oem/WORKS/DATA'

Traindata_frt = pd.read_csv('{0}/CheXpert-v1.0{1}/train_frt.csv'.format(path, img_type)) ###
Traindata_frt = Traindata_frt.sort_values('Path').reset_index(drop=True)
Traindata_frt.to_csv('{0}/CheXpert-v1.0{1}/train_frt.csv'.format(path, img_type), index = False) ###
Traindata_lat = pd.read_csv('{0}/CheXpert-v1.0{1}/train_lat.csv'.format(path, img_type)) ###
Traindata_lat = Traindata_lat.sort_values('Path').reset_index(drop=True)
Traindata_lat.to_csv('{0}/CheXpert-v1.0{1}/train_lat.csv'.format(path, img_type), index = False) ###

pathFileTrain_frt = '{0}/CheXpert-v1.0{1}/train_frt.csv'.format(path, img_type) ###
pathFileTrain_lat = '{0}/CheXpert-v1.0{1}/train_lat.csv'.format(path, img_type) ###
pathFileValid_frt = '{0}/CheXpert-v1.0{1}/valid_frt.csv'.format(path, img_type)
pathFileValid_lat = '{0}/CheXpert-v1.0{1}/valid_lat.csv'.format(path, img_type)
pathFileTest_frt = '{0}/CheXpert-v1.0{1}/test_frt.csv'.format(path, img_type)
pathFileTest_lat = '{0}/CheXpert-v1.0{1}/test_lat.csv'.format(path, img_type)
pathFileTest_agg = '{0}/CheXpert-v1.0{1}/test_agg.csv'.format(path, img_type)



# Neural network parameters
nnIsTrained = cfg.pre_trained # if pre-trained by ImageNet

# Training settings
trBatchSize = cfg.batch_size # batch size
trMaxEpoch = cfg.epochs      # maximum number of epochs

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = cfg.imgtransResize

# Class names
nnClassCount = cfg.nnClassCount   # dimension of the output - 5: only competition obs.
class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]



######################
## Create a Dataset ##
######################
# Tranform data
transformList = []
transformList.append(transforms.Resize((imgtransResize, imgtransResize))) # 320
transformList.append(transforms.ToTensor())
transformSequence = transforms.Compose(transformList)

# Create a dataset
'''See 'materials.py' to check the class 'CheXpertDataSet'.'''
datasetTrain_frt = CheXpertDataSet(pathFileTrain_frt, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat = CheXpertDataSet(pathFileTrain_lat, nnClassCount, cfg.policy, transformSequence)
datasetValid_frt = CheXpertDataSet(pathFileValid_frt, nnClassCount, cfg.policy, transformSequence)
datasetValid_lat = CheXpertDataSet(pathFileValid_lat, nnClassCount, cfg.policy, transformSequence)
datasetTest_frt = CheXpertDataSet(pathFileTest_frt, nnClassCount, cfg.policy, transformSequence)
datasetTest_lat = CheXpertDataSet(pathFileTest_lat, nnClassCount, cfg.policy, transformSequence)
datasetTest_agg = CheXpertDataSet(pathFileTest_agg, nnClassCount, cfg.policy, transformSequence)

# Use subset of datasetTrain for training ###
train_num_frt = round(len(datasetTrain_frt) * cfg.train_ratio) # use subset of original training dataset
train_num_lat = round(len(datasetTrain_lat) * cfg.train_ratio) # use subset of original training dataset
datasetTrain_frt, _ = random_split(datasetTrain_frt, [train_num_frt, len(datasetTrain_frt) - train_num_frt])
datasetTrain_lat, _ = random_split(datasetTrain_lat, [train_num_lat, len(datasetTrain_lat) - train_num_lat])
print('<<< Data Information >>>')
print('Train data (frontal):', len(datasetTrain_frt))
print('Train data (lateral):', len(datasetTrain_lat))
print('Valid data (frontal):', len(datasetValid_frt))
print('Valid data (lateral):', len(datasetValid_lat))
print('Test data (frontal):', len(datasetTest_frt))
print('Test data (lateral):', len(datasetTest_lat))
print('Test data (study):', len(datasetTest_agg), '\n')

# Create DataLoaders
dataLoaderTrain_frt = DataLoader(dataset = datasetTrain_frt, batch_size = trBatchSize, 
                                 shuffle = True, num_workers = 2, pin_memory = True) ###
dataLoaderTrain_lat = DataLoader(dataset = datasetTrain_lat, batch_size = trBatchSize, 
                                 shuffle = True, num_workers = 2, pin_memory = True) ###
dataLoaderVal_frt = DataLoader(dataset = datasetValid_frt, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderVal_lat = DataLoader(dataset = datasetValid_lat, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTest_frt = DataLoader(dataset = datasetTest_frt, num_workers = 2, pin_memory = True)
dataLoaderTest_lat = DataLoader(dataset = datasetTest_lat, num_workers = 2, pin_memory = True)
dataLoaderTest_agg = DataLoader(dataset = datasetTest_agg, num_workers = 2, pin_memory = True)



#####################
## Train the Model ##
#####################
# Initialize and load the model
'''See 'materials.py' to check the class 'DenseNet121'.'''
PATH = args.output_path
if args.output_path[-1] != '/':
    PATH = PATH + '/'
else:
    PATH = PATH


if not os.path.exists(PATH): os.makedirs(PATH)


model_Teacher = DenseNet121(nnClassCount, nnIsTrained).cuda()
model_Teacher = torch.nn.DataParallel(model_Teacher).cuda()


model_Teacher, model_Teacher_num_frt, model_Teacher_num_frt_each, model_Teacher_num_lat_each, train_Teacher_valid_start_frt, train_Teacher_valid_end_frt, train_Teacher_time_frt, train_Teacher_valid_start_lat, train_Teacher_valid_end_lat, train_Teacher_time_lat = Train_Model(model_Teacher, dataLoaderTrain_frt, dataLoaderTrain_lat, dataLoaderVal_frt, dataLoaderVal_lat, class_names, nnClassCount, trMaxEpoch, PATH, cfg, "Teacher")


model_Student = CheXpertMobileNetV3(nnClassCount, nnIsTrained).cuda()
model_Student = torch.nn.DataParallel(model_Student).cuda()


model_Student, model_Student_num_frt, model_Student_num_frt_each, model_Student_num_lat_each, train_Student_valid_start_frt, train_Student_valid_end_frt, train_Student_time_frt, train_Student_valid_start_lat, train_Student_valid_end_lat, train_Student_time_lat = Train_Model(model_Student, dataLoaderTrain_frt, dataLoaderTrain_lat, dataLoaderVal_frt, dataLoaderVal_lat, class_names, nnClassCount, trMaxEpoch, PATH, cfg, "Student")



outGT_Teacher, outPRED_Teacher, aurocMean_Teacher, aurocIndividual_Teacher = Test_ROC(PATH, model_Teacher, dataLoaderTest_frt, dataLoaderTest_lat, nnClassCount, class_names, "Teacher", pathFileTest_frt, pathFileTest_lat)

outGT_Student, outPRED_Student, aurocMean_Student, aurocIndividual_Student = Test_ROC(PATH, model_Student, dataLoaderTest_frt, dataLoaderTest_lat, nnClassCount, class_names, "Student", pathFileTest_frt, pathFileTest_lat)






NModel_Student = CheXpertMobileNetV3(nnClassCount, nnIsTrained).cuda()
NModel_Student = torch.nn.DataParallel(NModel_Student).cuda()


train_valid_start_Distill = time.time()

##########################3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




NModel_Student, model_num_NewStudent, model_num__NewStudent_each, train_time_NewStudent  =  train_knowledge_distillation(model_Teacher, NModel_Student, dataLoaderTrain_frt, dataLoaderVal_frt, nnClassCount, trMaxEpoch, 0.001, 2, 0.25, 0.75, device, "New_Student", "frt", PATH)

NModel_Student, model_num_NewStudent, model_num__NewStudent_each, train_time_NewStudent  =  train_knowledge_distillation(model_Teacher, NModel_Student, dataLoaderTrain_lat, dataLoaderVal_lat, nnClassCount, trMaxEpoch, 0.001, 2, 0.25, 0.75, device, "New_Student", "lat", PATH)

train_valid_end_Distill = time.time()


outGT_NStudent, outPRED_NStudent, aurocMean_NStudent, aurocIndividual_NStudent = Test_ROC(PATH, NModel_Student, dataLoaderTest_frt, dataLoaderTest_lat, nnClassCount, class_names, "New_Student", pathFileTest_frt, pathFileTest_lat)

#########################
## Computational Stats ##
#########################
print('<<< Computational Stats Teacher >>>')
print(train_Teacher_time_frt.round(0), '/seconds per epoch. (frt)')
print('Total', round((train_Teacher_valid_end_frt - train_Teacher_valid_start_frt) / 60), 'minutes elapsed.')
print(train_Teacher_time_lat.round(0), '/seconds per epoch. (lat)')
print('Total', round((train_Teacher_valid_end_lat - train_Teacher_valid_start_lat) / 60), 'minutes elapsed.')

print('<<< Computational Stats Student >>>')
print(train_Student_time_frt.round(0), '/seconds per epoch. (frt)')
print('Total', round((train_Student_valid_end_frt - train_Student_valid_start_frt) / 60), 'minutes elapsed.')
print(train_Teacher_time_lat.round(0), '/seconds per epoch. (lat)')
print('Total', round((train_Student_valid_end_lat - train_Student_valid_start_lat) / 60), 'minutes elapsed.')


print(train_time_NewStudent.round(0), '/seconds per epoch. (lat)')
print('Total', round((train_valid_start_Distill - train_valid_start_Distill) / 60), 'minutes elapsed.')
