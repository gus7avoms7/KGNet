# Made by Gustavo Marques - Federal University of São Carlos
# gustavomarques@estudante.ufscar.br
# @gustavoms7

#%% Setup: Initialization

# Initial Setup and Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
import time as time

# To further improve your code, you can use these common libraries
from pprint import pprint
from numpy import matlib


timElapsed = time.time() # Timer to compute total elapsed time

# Plotting Setup
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


#%% Function - Convert Time


# Function to put the time variables more friendly
def convert(seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds) 


#%% Setup: Import DeepMIMO I1_2p4 Scenario


timSetup = time.time() # Timer to compute setup time

# You can find this, and other scenarios at https://deepmimo.net/ Also you can find explanation about inputs and outputs
import DeepMIMO

# Load the default parameters
parameters = DeepMIMO.default_params()

# Edit the parameters
parameters = { 'dataset_folder': r'C:\Users\gusta\Downloads\Scenarios',
               'scenario': 'I1_2p4', # Band 1
               'dynamic_settings': {'first_scene': 1, 'last_scene': 1},
               'num_paths': 5,
               'active_BS': np.array([1]),
               'user_row_first': 1,
               #'user_row_last': 502,             # For my specific application, I cannot insert randomness at the users, since
               #'row_subsampling': 0.8,           the different users actually represents a single user into different possible positions
               #'user_subsampling': 0.8383,       Then, for the same position I need a band1 (Bob to Alice) and band2 (Alice to Bob)
               'user_row_last': 10,              # You can use this to validate your code with just 201 users, before consider more users
               #'user_row_last': 300,
               'row_subsampling': 1,
               'user_subsampling': 1,
               'bs_antenna': {'shape': np.array([1, 1, 1]),
                              'spacing': 0.5,
                              'radiation_pattern': 'isotropic'
                              },
               'ue_antenna': {'shape': np.array([1, 1, 1]),
                              'spacing': 0.5,
                              'radiation_pattern': 'isotropic'
                              },
               'enable_BS2BS': 0,
               'OFDM_channels': 1,
               'OFDM': {'subcarriers': 64,
                        'subcarriers_limit': 64,
                        'subcarriers_sampling': 1,
                        'bandwidth': 0.5, 
                        'RX_filter': 0
                        }
               }

# Generate data
dataset1 = DeepMIMO.generate_data(parameters)


#%% Setup: Import DeepMIMO I1_2p5 Scenario


# You can find this, and other scenarios at https://deepmimo.net/ Also you can find explanation about inputs and outputs
import DeepMIMO

# Load the default parameters
parameters = DeepMIMO.default_params()

# Edit the parameters
parameters = { 'dataset_folder': r'C:\Users\gusta\Downloads\Scenarios',
               'scenario': 'I1_2p5', #band 2
               'dynamic_settings': {'first_scene': 1, 'last_scene': 1},
               'num_paths': 5,
               'active_BS': np.array([1]),
               'user_row_first': 1,
               #'user_row_last': 502,             # For my specific application, I cannot insert randomness at the users, since
               #'row_subsampling': 0.8,           the different users actually represents a single user into different possible positions
               #'user_subsampling': 0.8383,       Then, for the same position I need a band1 (Bob to Alice) and band2 (Alice to Bob)
               'user_row_last': 10,              # You can use this to validate your code with just 201 users, before consider more users
               #'user_row_last': 300,
               'row_subsampling': 1,
               'user_subsampling': 1,
               'bs_antenna': {'shape': np.array([1, 1, 1]),
                              'spacing': 0.5,
                              'radiation_pattern': 'isotropic'
                              },
               'ue_antenna': {'shape': np.array([1, 1, 1]),
                              'spacing': 0.5,
                              'radiation_pattern': 'isotropic'
                              },
               'enable_BS2BS': 0,
               'OFDM_channels': 1,
               'OFDM': {'subcarriers': 64,
                        'subcarriers_limit': 64,
                        'subcarriers_sampling': 1,
                        'bandwidth': 0.5, 
                        'RX_filter': 0
                        }
               }

# Generate data
dataset2 = DeepMIMO.generate_data(parameters)


#%% Setup: DataSet Storage


# Storing the dataset in a simpler variable: A Matrix
ds1 = dataset1[0]['user']['channel']
ds2 = dataset2[0]['user']['channel']


#%% Setup: DataSet Shape


# Discovering the dataset shape
print(len(ds1),len(ds1[0]),len(ds1[0,0]),len(ds1[0,0,0]))
print(len(ds2),len(ds2[0]),len(ds2[0,0]),len(ds2[0,0,0]))


#%% Setup: DataSet Reshaping


# It is clear that there is variation between  the number of 
# users and the subcarrier frequency (first and fourth dimensions)
# So let's make this matrix more friendly. Let's transform this 4D matrix
# into a 2D matrix - User x Subcarrier i.e. the rows of these arrays
# represent the users and the columns represent the subcarriers 
ds1=np.reshape(ds1,(len(ds1),len(ds1[0,0,0])))
ds2=np.reshape(ds2,(len(ds2),len(ds2[0,0,0])))
print(len(ds1),len(ds1[0]))
print(len(ds2),len(ds2[0]))

#%% Timer - Setup


setupTime = time.time() - timSetup


#%% PreProcessing: Training/Testing Setup


# Defining the training and testing datasets (Zhang Section V. A)
timProc = time.time() # Timer to compute the preprocessing time

# Defining the percentage of training dataset
trpercent = 0.8
trsize = round(trpercent*len(ds1))

# Generating the random indexes of the users to be used as training dataset
trindices = sorted(np.random.choice(len(ds1), trsize, replace=False), key=int)
#print(trindices) #You can uncomment this to analyze the indexes generated

# Creating the training and testing datasets
# trpercent*TotalUsers = Number of Users used to train the neural network
# (1-trpercent)*TotalUsers = Number of User used to test the neural network
# Where ds1 corresponds to the channel matrix H1
# Where ds2 corresponds to the channel matrix H2
ds1tr=ds1[trindices]
ds1te=np.delete(ds1,[trindices],0)
ds2tr=ds2[trindices]
ds2te=np.delete(ds2,[trindices],0)


#%% PreProcessing: Gaussian Noise


# Adding Additive White Gaussian Noise to testing dataset
# You can find the source code that inspired this one at 
# (https://www.rfwireless-world.com/source-code/Python/AWGN-python-script.html)

target_snr_db = 0

# Getting the signal Mean Power and dB
ds1te_coefficient = ds1te
ds1te_gain = np.absolute(ds1te_coefficient) ** 2
ds1te_db = 10 * np.log10(ds1te_gain)

ds2te_coefficient = ds2te
ds2te_gain = np.absolute(ds2te_coefficient) ** 2
ds2te_db = 10 * np.log10(ds2te_gain)

# Calculate average signal power and convert to dB
np.mean(np.absolute(ds1te), axis=1)
sig1_avg_gain = np.mean(ds1te_gain, axis=1)
sig1_avg_db = 10 * np.log10(sig1_avg_gain)

sig2_avg_gain = np.mean(ds2te_gain, axis=1)
sig2_avg_db = 10 * np.log10(sig2_avg_gain)

# Calculate noise and convert it to gain
noise1_avg_db = sig1_avg_db - target_snr_db
noise2_avg_db = sig2_avg_db - target_snr_db

#print("Average Noise power in dB = ", noise_avg_db)
noise1_avg_gain = 10 ** (noise1_avg_db / 10)
noise2_avg_gain = 10 ** (noise2_avg_db / 10)

# Generate samples of white noise
mean_noise = 0
noise1_coefficient = np.random.normal(mean_noise, np.sqrt(noise1_avg_gain), len(ds1te_gain))
noise2_coefficient = np.random.normal(mean_noise, np.sqrt(noise2_avg_gain), len(ds2te_gain))

# Add noise to original sine waveform signal
ds1te_noise = ds1te_coefficient + noise1_coefficient.reshape(-1,1)
ds2te_noise = ds2te_coefficient + noise2_coefficient.reshape(-1,1)

# Uncomment to apply:
ds1te = ds1te_noise # In order to keep all code working, it is needed to certify that ds1te_noise has the same length of orignir ds1te
ds2te = ds2te_noise # In order to keep all code working, it is needed to certify that ds2te_noise has the same length of orignir ds2te


#%% PreProcessing: Dataset Realization


# Realizing the training and testing datasets (Zhang Section V. B)
# Separating the real and imaginary parts, we will transform the Users x L array 
# in Users x 2L array (where L represents the subcarriers number)
ds1tr_realized=np.zeros([len(ds1tr),2*len(ds1tr[0])])
ds1tr_realized[:,0:len(ds1tr[0])]=np.real(ds1tr)
ds1tr_realized[:,len(ds1tr[0]):2*len(ds1tr[0])]=np.imag(ds1tr)

ds2tr_realized=np.zeros([len(ds2tr),2*len(ds2tr[0])])
ds2tr_realized[:,0:len(ds2tr[0])]=np.real(ds2tr)
ds2tr_realized[:,len(ds2tr[0]):2*len(ds2tr[0])]=np.imag(ds2tr)

ds1te_realized=np.zeros([len(ds1te),2*len(ds1te[0])])
ds1te_realized[:,0:len(ds1te[0])]=np.real(ds1te)
ds1te_realized[:,len(ds1te[0]):2*len(ds1te[0])]=np.imag(ds1te)

ds2te_realized=np.zeros([len(ds2te),2*len(ds2te[0])])
ds2te_realized[:,0:len(ds2te[0])]=np.real(ds2te)
ds2te_realized[:,len(ds2te[0]):2*len(ds2te[0])]=np.imag(ds2te)


#%% TestPoint - Realization


#print(ds1tr_realized.shape)
#print(ds1tr_realized[0,0],ds1tr_realized[0,64])
#print(ds1tr[0][0])

# The test then proves that the values were divided as follows:
# The initial array (ds1) has dimensions of NumUsers x NumSubCarriers, for userRowLast = 10: 2010 x 64;
# The array realized (ds1_realized) has dimensions NumUsers x 2*NumSubCarriers, for userRowLast = 10: 2010 x 128;
# Where, the indexes 0 to len(ds1)-1 indicate the real values and the indexes len(ds1) to 2*len(ds1)-1 the imaginary ones
# For userRowLast = 10: the indexes 0 to 63 indicate the real values and the indexes 64 to 127 the imaginary ones


#%% PreProcessing: Dataset Normalization


# Normalizing the training and testing datasets (Zhang Section V. B)
ds1tr_realized_maxValue = ds1tr_realized.max(axis=0)
ds1tr_realized_minValue = ds1tr_realized.min(axis=0)
ds1te_realized_maxValue = np.matlib.repmat(ds1tr_realized_maxValue,len(ds1te_realized),1)
ds1te_realized_minValue = np.matlib.repmat(ds1tr_realized_minValue,len(ds1te_realized),1)
ds1tr_realized_maxValue = np.matlib.repmat(ds1tr_realized_maxValue,len(ds1tr_realized),1)
ds1tr_realized_minValue = np.matlib.repmat(ds1tr_realized_minValue,len(ds1tr_realized),1)
x1tr = (ds1tr_realized-ds1tr_realized_minValue)/(ds1tr_realized_maxValue-ds1tr_realized_minValue)
x1te = (ds1te_realized-ds1te_realized_minValue)/(ds1te_realized_maxValue-ds1te_realized_minValue)

ds2tr_realized_maxValue = ds2tr_realized.max(axis=0)
ds2tr_realized_minValue = ds2tr_realized.min(axis=0)
ds2te_realized_maxValue = np.matlib.repmat(ds2tr_realized_maxValue,len(ds2te_realized),1)
ds2te_realized_minValue = np.matlib.repmat(ds2tr_realized_minValue,len(ds2te_realized),1)
ds2tr_realized_maxValue = np.matlib.repmat(ds2tr_realized_maxValue,len(ds2tr_realized),1)
ds2tr_realized_minValue = np.matlib.repmat(ds2tr_realized_minValue,len(ds2tr_realized),1)
x2tr = (ds2tr_realized-ds2tr_realized_minValue)/(ds2tr_realized_maxValue-ds2tr_realized_minValue)
x2te = (ds2te_realized-ds2te_realized_minValue)/(ds2te_realized_maxValue-ds2te_realized_minValue)


#%% TestPoint - Normalization


# Showing that the replication of the array works 
#print(ds1tr_realized_maxValue)
#print(ds1tr_realized_maxValue.shape)

# Showing the datasets after the preprocessing steps
#print(x1tr)
#print(x2tr)
#print(x1te)
#print(x2te)
#print(x1tr.shape)
#print(x2tr.shape)
#print(x1te.shape)
#print(x2te.shape)


#%% Timer - PreProcessing


proccessingTime = time.time() - timProc


#%% Comment - We have the preprocessing done!


# Okay, we've done all the datasets preprocessing. Now we need to build a neural network 
# able to receive x1 as an input and somehow predict an x2_approximate as close as possible to x2.


#%% Neural Network: Construction


# A good way to understand neural networks is to participate the Kaggle Intro to DeepLearning course,
# available for free at https://www.kaggle.com/learn/intro-to-deep-learning?rvi=1

# Setup System
from tensorflow import keras
from tensorflow.keras import layers

# To further improve your code, you can use these common libraries
from tensorflow.keras import callbacks
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

# Defining needed variables to Neural Network training and testing processes
X_train = x1tr
X_valid = x1te
y_train = x2tr
y_valid = x2te

input_shape = len(X_train[0])
print("Input shape: {}".format(input_shape))


#%% Neural Network: Model Setup


# Defining the model of the Neural Network, i.e., defining the number of 
# layers, neurons, activation functions, etc.
model = keras.Sequential([
    layers.Dense(len(X_train), activation='relu', input_shape=[input_shape]),
    layers.Dense(512, activation='relu'),    
    layers.Dense(1024, activation='relu'),    
    layers.Dense(1024, activation='relu'),    
    layers.Dense(512, activation='relu'),    
    layers.Dense(input_shape,activation='sigmoid'),
])


#%% Neural Network: Optimizer Setup


# Defining the optimizer used to train the Neural Network
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer = opt,
    loss = 'mse',
)


#%% TestPoint - Neural Network Inputs Shapes


#print(X_train.shape)
#print(y_train.shape)


#%% Neural Network: Training and Validation


timTrain = time.time() # Timer to compute the training time
history = model.fit(
    X_train,y_train,
    validation_data=(X_valid, y_valid),
    batch_size = 128,
    epochs = 60,
    verbose = 1,
)


#%% Timer - Neural Network Training


trainingTime = time.time() - timTrain


#%% Results: Plotting and Saving History (Loss)


history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss']].plot();
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,5));
#plt.savefig('FigX_variationScenario_noValidationLossCurve.svg')


#%% Results: Plotting and Saving History (Loss and validationLoss)


history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,5));
#plt.savefig('FigX_variationScenario_withValidationLoss.svg')


#%% Results: Saving History to MatLab


history.history
from scipy.io import savemat
mdic = {"Loss": history.history}
#savemat("FigX_hist_variationScenario.mat", mdic)


#%% Results: Getting and Saving a Prediction


timPredict = time.time() # Timer to compute the prediction time
x2approx = model.predict(X_valid)
mdic = {"X_valid": X_valid, "y_valid": y_valid,"X_train": X_train, "y_train": y_train, "x2approx": x2approx}
#savemat("FigX_result_variationScenario.mat", mdic)

#%% TestPoint - Prediction


#print(y_valid)
#print(x2approx)


#%% Timer - Prediction


predictTime = time.time()-timPredict


#%% Function - Quantization


# Quantization Algorithm GDQG
from statistics import NormalDist
def quantization(x,xi,K): 
    L = int(len(x[0])/2); # Number of subcarriers
    NBits = 2*L;
    NUsers = len(x);
    K = int(K);
    intervals = np.zeros([K,2]);
    intervalDecValues = np.linspace(0,K-1,K);
    mu = 1/(2*L) * np.sum(x,axis=1);
    sig2 = 1/(2*L-1) * np.sum((x-mu.reshape(-1,1))**2,axis=1);
    Q = np.ones(x.shape)*(-1);
    for i in range(NUsers):
        intervals[0] = [0,NormalDist(mu=mu[i], sigma=sig2[i]).inv_cdf(1/K-xi)];
        for k in range(2,K):
            intervals[k-1]=[NormalDist(mu=mu[i], sigma=sig2[i]).inv_cdf((k-1)/K+xi),NormalDist(mu=mu[i], sigma=sig2[i]).inv_cdf(k/K-xi)];
        intervals[K-1] = [NormalDist(mu=mu[i], sigma=sig2[i]).inv_cdf((K-1)/K+xi),1];
        for j in range(NBits):
            for k in range(K):
                if x[i,j] > intervals[k,0] and x[i,j] < intervals[k,1]:
                    Q[i,j]=intervalDecValues[k];                    
    return [Q,mu,sig2,intervals]


#%% Quantization: Saving Result


timQuant = time.time() # Timer to compute the quantization time

#K = 2;
#xi = 0.3;
K = 8;
xi = 0.2/(2*(K-1));

Qa_aux=quantization(x2approx,xi,K)
Qb_aux=quantization(y_valid,xi,K)

Qa = Qa_aux[0]
Qb = Qb_aux[0]

Qa = [row for row in Qa]
Qb = [row for row in Qb]

NSubCarriers = len(Qa[0])/2
KER_users = np.zeros(len(Qa))
KGR_users = np.zeros(len(Qa))

for i in range(len(Qa)):
    errIndexA = np.argwhere(Qa[i] == -1)
    errIndexB = np.argwhere(Qb[i] == -1)
    errIndex = np.union1d(errIndexA,errIndexB)
    mask = np.ones(len(Qa[i]),dtype=bool)
    mask[errIndex] = False
    Qa[i] = Qa[i][mask]
    Qb[i] = Qb[i][mask]
    KER_users[i] = np.count_nonzero(Qa[i] != Qb[i])/len(Qa[i])
    KGR_users[i] = len(Qa[i])/NSubCarriers
    
KER = np.mean(KER_users)
KGR = np.mean(KGR_users)

mdic = {"Qa_aux": Qa_aux, "Qb_aux": Qb_aux,"KER": KER, "KGR": KGR}
#savemat("FigX_quantized_variationScenario.mat", mdic)


#%% Timer - Quantization


QuantizationTime = time.time()-timQuant


#%% Timer - Elapsed


elapsed = time.time() - timElapsed
print('Elapsed Time',convert(elapsed),'(%.1f s)' % elapsed) 
print('Setup Time',convert(setupTime),'(%.1f s)' % setupTime)
print('PreProcessing Time',convert(proccessingTime),'(%.1f s)' % proccessingTime)
print('Training Time',convert(trainingTime),'(%.1f s)' % trainingTime)
print('Prediction Time',convert(predictTime),'(%.1f s)' % predictTime)
print('Quantization Time',convert(QuantizationTime),'(%.1f s)' % QuantizationTime)

