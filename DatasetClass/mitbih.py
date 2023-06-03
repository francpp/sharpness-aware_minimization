import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from utilities.cutout import Cutout

class MitBih:
    def __init__(self, batch_size, threads):
        
        self.train_set = MitBihSubset(train=True)
        self.test_set = MitBihSubset(train=False)
        
        self.train = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('0', '1', '2', '3', '4')


class MitBihSubset(Dataset):
    def __init__(self, train=True):
        
        if train==True:
            try:
                mitbih_dataset = pd.read_csv('DatasetClass/mitbih/mitbih_train.csv')
            except:
                mitbih_dataset = pd.read_csv('../DatasetClass/mitbih/mitbih_train.csv')
            start_point_to_balance = 71829
        
        else:
            try:
                mitbih_dataset = pd.read_csv('DatasetClass/mitbih/mitbih_test.csv')
            except:
                mitbih_dataset = pd.read_csv('../DatasetClass/mitbih/mitbih_test.csv')
            start_point_to_balance = 17956
        
        x = mitbih_dataset.iloc[start_point_to_balance:,:-1].values
        y = mitbih_dataset.iloc[start_point_to_balance:,-1:].astype(dtype=int).values
        
        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        x = self.X[index].unsqueeze(1)
        y = self.Y[index]
        return x, y


"""
class MyTestDataLoader():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size 
        file_out_test = pd.read_csv('mitbih_test.csv')
        
        x_test = file_out_test.iloc[:,:-1].values
        y_test = file_out_test.iloc[:,-1:].astype(dtype=int).values   
 
        test_set= EcgMitBih(x = x_test, y= y_test) 
        self.dataLoader= DataLoader(test_set, batch_size=self.batch_size, shuffle=True,  ) 

    def getDataLoader(self): 
        return self.dataLoader

class myDataLoader():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        file_out_train = pd.read_csv('mitbih_train.csv') 

        x_train = file_out_train.iloc[:,:-1].values
        y_train = file_out_train.iloc[:,-1:].astype(dtype=int).values 
        x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.15 )  

        #print("x_train shape on batch size =  " + str(x_train.shape ))
        #print('x_val shape on batch size =  ' + str(x_val.shape))
        #print('y_train shape on batch size =  '+ str(y_train.shape ))
        #print('y_val shape on batch size =  ' + str( y_val.shape) )

        train_set= EcgDataset(x= x_train, y= y_train) 

        val_set= EcgDataset(x= x_val, y= y_val) 

        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,  ),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True,  )
        }
        self.dataloaders = dataloaders
        

    def getDataLoader(self): 
        return self.dataloaders
"""