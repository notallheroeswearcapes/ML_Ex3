# Machine Learning - Exercise 3 - Group 9

### Group Members
* Matthias Eder, 01624856
* Benedikt Hämmerle, 01352108
* Karl Eriksson, 12005817

## Description

We stick to the official Python packaging guide (https://packaging.python.org/tutorials/packaging-projects/) to create 
the setup.py file  
We use the package 'click' to create a command-line interface (https://click.palletsprojects.com/en/7.x/).
The datasets CIFAR-10 and FashionMNIST build the basis of our analysis.
We use MLP, kNN and Random Forest as algorithms from scikit-learn.

step 1: train the model
* option: the dataset to use, one of "FashionMNIST" or "CIFAR-10" [B]
* import dataset [B]
* run feature representation [B]
* run feature extraction [K]
* run visual bag of words [K]
* --> output: just progress bars for each step? [B, K]

step 2: [M]
* option: a classifier algorithm, one of "MLP", "kNN" or "RandomForest" 
* output: the parameters of the classifiers
* run classifier on previously tested best parameters (mention in report)
* save results
* output: export a confusion matrix of all classes, include runtime and parameter settings here as well
* output: notify user of successfully exported confusion matrix, accuracy score and the runtime 

step 3: 
* option: a CNN architecture, one of "ResNet 50" or "???" [B, K]
* run the CNN
* output: export a confusion matrix of all classes, include runtime and parameter settings here as well
* output: notify user of successfully exported confusion matrix, accuracy score and the runtime 

step 4: 
* option: none, name "--evaluate"
* compare classifier and CNN
* output: export both confusion matrices side-to-side, include runtimes and accuracy scores
* output: a table with both runtimes and accuracy scores (ASCII table in command line output)

optional step:
* use data augmentation on the CNN chosen before
* run the CNN
* output: export a confusion matrix of all classes, include runtime and parameter settings here as well
* output: notify user of successfully exported confusion matrix, accuracy score and the runtime 

## Packages

| Package      	| Version 	|
|--------------	|---------	|
| Python       	| 3.8.0   	|
| click        	| 7.1.2 	|
| setuptools   	| 53.0.0 	|
| wheel        	| 0.36.2 	|

## Guide
creating the virtualenv https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env

save the dependencies: `pip freeze > requirements.txt`, delete the egg (-e) and make sure to have the virtualenv activated

install dependencies: `pip install -r requirements.txt`

install the CLI: `pip install --editable .` from the project root directory
