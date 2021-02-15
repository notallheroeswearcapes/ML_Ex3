import numpy as np
import pandas as pd
import click
import os
from tensorflow.keras.datasets import cifar10
from PIL import Image
from deeplearn import io


class FeatureExtractor:

    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        
        
    def Extractor(self):
        
        
        self.load_data()
        self.train_data = self.train_data[0:100]
        self.test_data = self.test_data[0:10]
        feature_data = []
        features_df = pd.DataFrame() 
                
        for fileName in self.test_data:
            
            fileImage = Image.open(fileName)
                        
            # ensure that all images are RGB
            fileImage = fileImage.convert('RGB')  
            
            # extract feature to 1D array 
            features=fileImage.histogram()
            
            if (len(features) == 768): # check if feature array is what we expect; else discard
                
                #transform to array and then to df
                feature_data = np.array(feature_data)
                feature_data = pd.DataFrame(feature_data.reshape(-1, len(feature_data)))
                feature_data.insert(0, "", fileName) 
                
            #append to dataframe for export
            feature_df = feature_df.append(feature_data, ignore_index=True)
                
        #export and finish
        feature_df.to_csv("data1.csv")
        click.echo("All features extracted!")        
                        
        
    def load_data(self):
        self.train_data = io.import_numpy('data/CIFAR-10_train_data.npy')
        self.test_data = io.import_numpy('data/CIFAR-10_test_data.npy')
        click.echo("DONE Reading data.")