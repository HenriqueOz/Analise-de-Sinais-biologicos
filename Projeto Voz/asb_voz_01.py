# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:49:13 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de Sistemas Biologicos
Aula: Projeto de reconhecimento de voz
Exercício: 1) Classificadores
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
import wave
import pandas as pd

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

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# Open the RAW data files
files  = askopenfilenames(title = "Select .wav files",
                                   filetypes = (("Wav File","*.wav"),("all files","*.*")))
# Open the MOEN file for mnapping the runs parameters
moen_file_path = askopenfilename(initialfile = "moen_file.csv",
                                 title = "Select MOEN file",
                                 filetypes = (("CSV file","*.csv"),("all files","*.*")))
skipped = 0

moen_data = pd.read_csv(moen_file_path, sep = ',', header = 0, engine = 'python')

for file_name in files:

    file = wave.open(file_name)
    str_file_name = os.path.basename(file_name).replace(".wav","")

    if (str_file_name in moen_data["file_name"].values):
        print(str_file_name)
        print("Number of channels: " + str(file.getnchannels()))
        s_rate = file.getframerate()
        print("Sampling rate: " + str(s_rate))
        n_frames = file.getnframes()
        print("Number of frames: " + str(n_frames))
        print("Sample size: " + str(file.getsampwidth()))

        data_bin = file.readframes(-1)
        data = np.frombuffer(data_bin, np.int16)
        data.shape = -1,2
        data = data.T
        if (file.getnchannels() == 2):
            data = data[0,:] # Consider only one channel of the data if it stereo

        mask = envelope(data, s_rate, 100)
        data = data[mask]
        t = np.arange(0, data.shape[0]/float(s_rate), 1/s_rate)
        t, data = check_size(t, data)

        ds = 4 #Fator de downsampling

        plt.figure(figsize=(12,6))
        plt.plot(t,data, label = "Sinal no tempo - canal 0")
        plt.xlabel('Tempo (s)')
        plt.ylabel('Intensidade')
        plt.title('Sinal de ' + str_file_name + " no tempo")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname = "Sinal_no_tempo_" + str_file_name + ".jpeg", dpi = 600)
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

        plt.figure(figsize=(12,6))
        plt.plot(FRW, 20*np.log10(YRW),
                label='Janela retangular',
                linestyle = 'solid',
                linewidth = 1)
        plt.plot(FH, 20*np.log10(YH),
                label='Filtrado',
                linestyle = 'solid',
                linewidth = 1)
        plt.grid()
        plt.legend()
        plt.xlabel('Frequencia [Hz]')
        plt.ylabel('Intensidade [db]')
        plt.title('Sinal de ' + str_file_name + " no espaço da frequencia")
        plt.tight_layout()


    else:
        print(file_name + " was skipped")
        skipped += 1



