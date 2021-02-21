# Machine Learning - Exercise 3 - Group 9

### Group Members
* Matthias Eder, 01624856
* Benedikt Hämmerle, 01352108
* Karl Eriksson, 12005817

## Description

This repository contains the source code of our solution for the third exercise of the course _Machine Learning_ from TU Wien in the winter semester of 2020/2021. 
We have chosen the topic 2 and present the program `deeplearn` which uses different algorithms to classify image data. 
The integrated datasets are [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
First, colour histograms are created from the chosen dataset to represent features. Then we run a SIFT-based feature extractor on the data and create a visual bag of words.
After these preparation steps, the data is ready to be classified. 
This can either be done via three more traditional algorithms k-NN, MLP and RandomForest or the two deep learning architectures Resnet-50 or Keras-CNN which both use a Convolutional Neural Network.
The resulting classification predictions can be evaluated and compared.

We use the package [click](https://click.palletsprojects.com/en/7.x/) to create a command-line interface. 
The implementations of the traditional classifiers are taken from [sklearn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning).
As the training of the CNN models is computationally expensive, we ran this in a Google Colab and provide only the pre-trained models here. 
These files need to be downloaded separately and placed inside this project. As these files are very large and exceed the file size limit on TUWEL, we provide them on [Google Drive](https://drive.google.com/file/d/1zbqtcNQQHAjN6txrqvQKYazcd_FYMSX0/view?usp=sharing).
The raw data of CIFAR-10 and Fashion-MNIST does not need to be downloaded manually as they are fetched from [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/datasets).
Classification predictions are evaluated regarding runtime and accuracy. 
Please note that the runtimes given for the CNNs are only calculated for the prediction. The time for training is not included as it would not make the results comparable.

Disclaimer: We only guarantee that this project will run on a machine with Windows 10. 

## Installation Guide (for Windows)
1. make sure to have at least Python 3.8, pip 21.0.1 and virtualenv installed on your system
2. clone this repository in a new directory with `git clone https://github.com/notallheroeswearcapes/ML_Ex3.git`
3. navigate to the project root and create a virtual environment either by following [this tutorial](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env) for PyCharm or by running `virtualenv venv` in the command line
4. activate the virtual environment by running `venv\scripts\activate` (single backslashes)
5. install the dependencies of our project by running `pip install -r requirements.txt` (it might be necessary to do this from a conda prompt)
6. download the additional files provided on [Google Drive](https://drive.google.com/file/d/1zbqtcNQQHAjN6txrqvQKYazcd_FYMSX0/view?usp=sharing) containing the pre-trained CNN models and place the folder `trained_models` inside the project root directory
7. install our CLI by running `pip install .`
8. now you should be able to run our project by running `deeplearn --help`

Note: We recommend running everything from an Anaconda prompt.

## Instructions for `deeplearn`

`deeplearn --help` can be used to obtain help.

The base command `deeplearn -d dataset` fetches the raw data, creates the feature representation and extraction files and exports them to the folder `data` in the project root directory.
The _dataset_ option must be specified as either 'CIFAR-10' or 'Fashion-MNIST'.

The three commands `classify`, `cnn` and `evaluate` can be called on top of the base command. The option `--help` can be called on top of these commands as well.

The command `deeplearn -d dataset classify -a algorithm` runs a classification on the specified _dataset_ with the specified _algorithm_. 
The _algorithm_ option must be specified as either 'k-NN', 'MLP' or 'RandomForest'. The algorithm will be run on both the feature representation and extraction data as input.
This command outputs a confusion matrix of the predictions as a figure in each case, and an ASCII table with the respective runtimes and accuracy scores in the console. 

The command `deeplearn -d dataset cnn -a architecture` runs a CNN classification on the specified _dataset_ with the specified _architecture_.
The _architecture_ option must be specified as either 'Resnet-50' or 'Keras-CNN'. The CNN classification is run on the respective pre-trained model. This command also
outputs a confusion matrix of the prediction as a figure, and an ASCII table of the runtime and accuracy score.

The command `deeplearn -d dataset evaluate [-c classifier]` evaluates and compares the results of classifications on the specified _dataset_. 
The _classifier_ option must be specified as either 'k-NN', 'MLP', 'RandomForest', 'Resnet-50' or 'Keras-CNN'.
The `-c classifier` option can be declared multiple times but must be given at least once.
A specified classifier does not necessarily need to have been run before evaluation. In this case, `deeplearn` prompts if the classifier should be run now.
This command outputs an ASCII table of the respective runtimes and accuracy scores of the specified classifiers, and also a figure with all confusion matrices. 
However, this figure might scale badly with many classifiers given. 

Note: Please save the confusion matrix figures if you want to take a look at them again. The program will only continue once the figures are closed.

## License

    MIT License
    Copyright (c) 2021 Eder, Eriksson and Haemmerle
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Contact

Matthias Eder: eder.matthias1@gmail.com

Repository: https://github.com/notallheroeswearcapes/ML_Ex3

## Dependencies

Refer to build file `requirements.txt`.

| Package      	| Version 	|
|--------------	|---------	|
| Python       	| 3.8.5   	|
| pip       	| 21.0.1  	|
| virtualenv    | 20.2.2    |
|absl-py|0.11.0
|asgiref|3.3.1
|astunparse|1.6.3
|cachetools|4.2.1
|certifi|2020.12.5
|chardet|4.0.0
|click|7.1.2
|cycler|0.10.0
|Django|3.1.6
|flatbuffers|1.12
|gast|0.3.3
|google-auth|1.26.1
|google-auth-oauthlib|0.4.2
|google-pasta|0.2.0
|grpcio|1.32.0
|h5py|2.10.0
|idna|2.10
|image|1.5.33
|joblib|1.0.1
|Keras|2.4.3
|Keras-Preprocessing|1.1.2
|kiwisolver|1.3.1
|Markdown|3.3.3
|matplotlib|3.2.0
|numpy|1.19.5
|oauthlib|3.1.0
|opencv-contrib-python|4.5.1.48
|opt-einsum|3.3.0
|Pillow|8.1.0
|prettytable|2.0.0
|protobuf|3.14.0
|pyasn1|0.4.8
|pyasn1-modules|0.2.8
|pyparsing|2.4.7
|python-dateutil|2.8.1
|pytz|2021.1
|PyYAML|5.4.1
|requests|2.25.1
|requests-oauthlib|1.3.0
|rsa|4.7
|scikit-learn|0.24.1
|scipy|1.6.0
|six|1.15.0
|sklearn|0.0
|sqlparse|0.4.1
|tensorboard|2.4.1
|tensorboard-plugin-wit|1.8.0
|tensorflow|2.4.1
|tensorflow-estimator|2.4.0
|termcolor|1.1.0
|threadpoolctl|2.1.0
|typing-extensions|3.7.4.3
|urllib3|1.26.3
|wcwidth|0.2.5
|Werkzeug|1.0.1
|wrapt|1.12.1
