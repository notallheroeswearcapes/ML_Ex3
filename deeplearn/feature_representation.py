import numpy as np
import click
import os
from tensorflow.keras.datasets import cifar10
from PIL import Image
from deeplearn import io


class FeatureRepresentation:


    def __init__(self):
        self.train_data = None
        self.test_data = None


    def FeatureRep(self):

        self.train_data = self.train_data
        self.test_data = self.test_data
        feature_data_train = []
        feature_data_test = []
        
        click.echo("Starting feature representation for train data")

        for fileName in self.train_data:

            fileImage = Image.open(fileName)

            # ensure that all images are RGB
            fileImage = fileImage.convert('RGB')

            # extract feature to 1D array
            features=fileImage.histogram()

            feature_data_train.append.append(features)
        
        click.echo("Finised feature representation for train data")
        click.echo("Starting feature representation for test data")
        
        for fileName in self.test_data:

            fileImage = Image.open(fileName)

            # ensure that all images are RGB
            fileImage = fileImage.convert('RGB')

            # extract feature to 1D array
            features=fileImage.histogram()

            feature_data_test.append.append(features)
            
        click.echo("Finised feature representation for test data")    
            
        train_data = np.array(feature_data_train)
        test_data = np.array(feature_data_test)

        save('./data/train_data.npy', train_data)
        save('./data/test_data.npy', test_data)
        
click.echo("All data saved")            

    def load_data(self):
        self.train_data = io.import_numpy('data/CIFAR-10_train_data.npy')
        self.test_data = io.import_numpy('data/CIFAR-10_test_data.npy')
        click.echo("DONE Reading data.")
