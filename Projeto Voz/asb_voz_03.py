# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:49:13 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de Sistemas Biologicos
Aula: Projeto de reconhecimento de voz
Exercício: 1) Classificadores
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
import wave
import pandas as pd
from librosa import lpc


from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
#deeplizard 04 and 05
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
# deeplizard 08

from sklearn.metrics import confusion_matrix
import itertools

def mag_of_FFT(signal,Fs):
    #plot half spectrum
    #signal is the signal
    #Fs Sample Rate
    SIGNAL = np.fft.fft(signal);
    n = SIGNAL.shape[0];
    Ny= Fs/2; #Nyquist frequency
    f = np.arange(start = 0, stop = Ny, step = Fs/n)
    mag_SIGNAL = abs(SIGNAL[0:int(SIGNAL.shape[0]/2)])/ n;
    return check_size(mag_SIGNAL, f)

def check_size (vector_a, vector_b):
    if (vector_a.shape[0] == vector_b.shape[0]):
        return vector_a, vector_b
    elif(vector_a.shape[0] > vector_b.shape[0]):
        return vector_a[:-1], vector_b
    else:
        return vector_a,vector_b[:-1]

def envelope (y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window = int(rate/10), min_periods = 1, center = True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def fallZeroCross(x):
    z = []
    for idx, data in enumerate(x[:-1]):
        if (x[idx] > 1E-5 and x[idx+1] <= 0) or (x[idx] >= 0 and x[idx+1] < -1E-5):
            if abs(x[idx]) < abs(x[idx+1]):
                z.append(idx)
            else:
                z.append(idx)
    return z

first_file = True
def export_files():

    global first_file
    global z
    if(first_file):
            data.to_csv(path_or_buf = "Full_file.csv",
                        sep = ',',
                        mode = 'w',
                        header = True,
                        index = False)
            first_file = False
    else:
            data.to_csv(path_or_buf = "Full_file.csv",
                        sep = ',',
                        mode = 'a',
                        header = False,
                        index = False)

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# Open the RAW data files
files  = askopenfilenames(initialfile = 'hos_vogal_e.wav',
                          title = "Select .wav files",
                                   filetypes = (("Wav File","*.wav"),("all files","*.*")))
# Open the MOEN file for mnapping the runs parameters
moen_file_path = askopenfilename(initialfile = "moen_file.csv",
                                 title = "Select MOEN file",
                                 filetypes = (("CSV file","*.csv"),("all files","*.*")))
skipped = 0

moen_data = pd.read_csv(moen_file_path, sep = ',', header = 0, engine = 'python')
results = []

train_labels = []
train_samples = []
output_pics = False

for file_name in files:

    file = wave.open(file_name)
    str_file_name = os.path.basename(file_name).replace(".wav","")

    if (str_file_name in moen_data["file_name"].values):
        vogal = moen_data[moen_data["file_name"] == str_file_name]["vogal"].values
        vogal_num = moen_data[moen_data["file_name"] == str_file_name]["vogal_num"].values
        print(str_file_name)
        print("Number of channels: " + str(file.getnchannels()))
        s_rate = file.getframerate()
        print("Sampling rate: " + str(s_rate))
        n_frames = file.getnframes()
        print("Number of frames: " + str(n_frames))
        print("Sample size: " + str(file.getsampwidth()))

        data_bin = file.readframes(-1)
        data = np.frombuffer(data_bin, np.int16)

        if (file.getnchannels() == 2):
            data.shape = -1,2
            data = data.T
            data = data[0,:] # Consider only one channel of the data if it stereo
        else:
            data.shape = -1,1
            data = data.T
            data = data[0,:]

        mask = envelope(data, s_rate, 100)
        data_or = data[mask]

        ds = 4 #Fator de downsampling
        limit = 1000
        for diviser in range(1,20):
            limit = int(5000 + limit)
            data = data_or[limit:limit+5000]

            if (len(data)==0):
                break

            t = np.arange(0, data.shape[0]/float(s_rate), 1/s_rate)
            t, data = check_size(t, data)

            if(output_pics):
                plt.figure(figsize=(12,6))
                plt.plot(t,data, label = "Sinal no tempo - canal 0")
                plt.xlabel('Tempo (s)')
                plt.ylabel('Intensidade')
                plt.title('Sinal de ' + str_file_name + " no tempo")
                plt.grid()
                plt.legend()
                plt.tight_layout()
                plt.savefig(fname = "Sinal_no_tempo_" + str_file_name + "_" + str(diviser) + ".jpeg", dpi = 600)
            yrw = data[::ds]

            ## FFT do dado com janela retangular
            YRW, FRW = mag_of_FFT(data, s_rate/ds)

            h = [1, -0.95]
            imp = np.zeros(yrw.shape[0])
            imp[0] = 1
            yp = signal.lfilter(h, 1, imp)

            ys = signal.lfilter(h, 1, yrw)
            hamming_window = np.hamming(yrw.shape[0])
            yh = np.multiply(ys,hamming_window)
            ## FFT do dado com janela de hamming e filtro pré-enfasê
            YH, FH = mag_of_FFT(yh, s_rate/ds)

            if(output_pics):
                plt.figure(figsize=(12,6))
                plt.plot(FRW, 20*np.log10(YRW),
                        label='Janela retangular',
                        linestyle = 'solid',
                        linewidth = 1)
                plt.plot(FH, 20*np.log10(YH),
                        label='Filtrado',
                        linestyle = 'solid',
                        linewidth = 1)


            ## Classificadores #02
            A = lpc(yh,16)
            imp = np.zeros(yh.shape[0])
            imp[0] = 1
            lpc_x = signal.lfilter(np.atleast_1d([1]), A, imp)
            LPC_X, F_LPC_X = mag_of_FFT(4*lpc_x, s_rate/ds)

            if(output_pics):
                plt.plot(F_LPC_X, 20*np.log10(LPC_X),
                        label='LPC',
                        linestyle = 'solid',
                        linewidth = 1)


            X = abs(np.fft.fft(lpc_x)) # Cálculo do módulo da FFT (espectro)
            d = np.diff(X[0:int(X.shape[0]/2)],1)  # Derivada do módulo da FFT
            z = fallZeroCross(d) # Encontra os picos: Formantes

            if(output_pics):
                plt.plot(F_LPC_X[z], 35*np.ones(len(z)),
                        label='Formantes',
                        linestyle = 'none',
                        marker = 'd',
                        linewidth = 1)
                plt.grid()
                plt.legend()
                plt.xlabel('Frequencia [Hz]')
                plt.ylabel('Intensidade [db]')
                plt.title('Sinal de ' + str_file_name + " no espaço da frequencia")
                plt.tight_layout()
                plt.savefig(fname = "Espectro_" + str_file_name + "_" + str(diviser) + ".jpeg", dpi = 600)
                plt.close("all")
            print("Vogal " + str(vogal_num[0]) + ", formantes: " + str(F_LPC_X[z]))

            for idx, formante in enumerate(z):
                results.append([str_file_name, vogal[0], F_LPC_X[formante], idx, len(z), diviser])
                # train_labels.append(vogal_num[0])
                # train_samples.append(F_LPC_X[formante])
                if (idx == 2):
                    break
            train_labels.append(vogal_num[0])
            train_samples.append([F_LPC_X[z[0]],F_LPC_X[z[1]],F_LPC_X[z[2]]])
    else:
        print(file_name + " was skipped")
        skipped += 1


f = open("results.csv",'w', newline='')

with f:
    writer = csv.writer(f)
    writer.writerows(results)


#https://en.wikipedia.org/wiki/IPA_vowel_chart_with_audio
#https://www.google.com/search?q=formant+frequencies+of+vowels&sxsrf=ALeKk02tX1e6mGU28GoqNO2OIHrXAs5BoA:1605477934549&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiqze6vx4XtAhUkDrkGHfHHDcIQ_AUoAXoECBcQAw&biw=1536&bih=722#imgrc=ZHvqT8L_uJL1WM

# %% Parte 03 do exercicio: rede neural para identificar as vogais

# converter
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)
divisao_train_test = int(train_labels.shape[0]*0.2)
train_labels_ =  train_labels[:-divisao_train_test]
test_labels_ =   train_labels[-divisao_train_test:]

#funciona como a função map() do arduino: mapeia de 0 a 1 considerando os limites dos dados de entrada
scaler = MinMaxScaler(feature_range = (0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
# scaled_train_labels = scaler.fit_transform(train_labels.reshape(-1,1))

scaled_train_samples = np.column_stack((scaled_train_samples[0::3],
                                        scaled_train_samples[1::3],
                                        scaled_train_samples[2::3]))
scaled_train_samples_ = scaled_train_samples[:-divisao_train_test:,:]
scaled_test_samples =  scaled_train_samples[-divisao_train_test:,:]
# %% Deeplizard 04 - Instanciating the network

# tem um bloco de código para inicializar a GPU, mas não estou usando no momento

model = Sequential([
    Dense(units = 10, input_shape = (3,), activation = 'relu'),
    # Dense(units = 10, activation = 'relu'),
    Dense(units = 5, activation = 'softmax')])

# model.summary() display a summary of the model so far

#  Deeplizard 05 - training the network

model.compile(optimizer = Adam(learning_rate = 0.01),
              loss = 'sparse_categorical_crossentropy',
              metrics = ["accuracy"])

model.fit(x = scaled_train_samples_,
          y = train_labels_,
          batch_size = 10,
          # validation_split = 0.05,
          epochs = 150,
          shuffle = True,
          verbose = 2)

model.summary()
# %% Predict model
predictions = model.predict(x = scaled_test_samples,
                            batch_size = 10,
                            verbose = 0)
print(predictions)

rounded_predictions = np.argmax(predictions, axis = 1)
print(rounded_predictions)


""" A matrix de confusão mostra como o modelo está se saindo considerando que temos etiquetas para nossas amostras de teste.
    A função confusion_matrix basicamente compara os índices dos nós com as etiquetas conhecidas - por isso é bom usar etiquetas numeradas a partir de 0 e de um em um.
"""

cm = confusion_matrix(y_true = test_labels_, y_pred = rounded_predictions)
cm_plot_labels = ['a','e','i','o','u']

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = "Confusion Matrix",
                          cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normlization")
    print(cm)
    thresh = cm.max()/2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i,j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel("Vogal verdadeira")
        plt.xlabel("Vogal predizida")

plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = "Matrix de confusão")
