# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:23:05 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de Sinais Biológicos
Aula: 03?
Exercício: revisão processamento de sinais digitais
"""

import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# Exercício 1

def mag_of_plotFFT(signal,Fs):
    #plot half spectrum
    #signal is the signal
    #Fs Sample Rate
    SIGNAL = np.fft.fft(signal);
    n = SIGNAL.shape[0];
    Ny= Fs/2; #Nyquist frequency
    f = np.arange(start = 0, stop = Ny, step = Fs/n)
    mag_SIGNAL = abs(SIGNAL[0:int(SIGNAL.shape[0]/2)])/ n;
    plt.figure(figsize = (8,4))
    plt.plot(f,mag_SIGNAL)
    plt.ylabel("Magnitude")
    plt.xlabel("Frequencia")
    plt.title("Frequencias")

fo = 440          #Fundamental frequency
To = 1/fo
Fs =  44100     # Sample rate
Ts = 1/Fs
A = 1.0         #Amplitude
t = np.arange(start = 0, stop = 3, step = Ts)

x = A*np.sin(2*np.pi*fo*t)

plt.figure(figsize = (8,4))
plt.plot(t,x)
plt.ylabel("Amplitude")
plt.xlabel("Tempo [s]")
plt.title("Sinal no tempo - seno 440hz")

mag_of_plotFFT(x, Fs)

x2 = A*np.sin(2*np.pi*fo*t) + A/2*np.sin(2*np.pi*4*fo*t) + A/3*np.sin(2*np.pi*11.36*fo*t)

plt.figure(figsize = (8,4))
plt.plot(t,x)
plt.ylabel("Amplitude")
plt.xlabel("Tempo [s]")
plt.title("Sinal no tempo - soma de senos")
mag_of_plotFFT(x2, Fs)

for i in range(1,5):
     plt.figure(i)
     plt.savefig(fname = "fig " + str(i) + ".jpeg", dpi = 600)
