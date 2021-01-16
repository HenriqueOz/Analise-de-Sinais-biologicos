# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:28:25 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de Sinais Biológicos
Aula 3: Revise o assunto convolução.
Exercício 3: Convolução/média móvel
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def mag_of_FFT(signal,Fs):
    #plot half spectrum
    #signal is the signal
    #Fs Sample Rate
    SIGNAL = np.fft.fft(signal);
    n = SIGNAL.shape[0];
    Ny= Fs/2; #Nyquist frequency
    f = np.arange(start = 0, stop = Ny, step = Fs/n)
    mag_SIGNAL = abs(SIGNAL[0:int(SIGNAL.shape[0]/2)])/ n;
    return mag_SIGNAL, f

# %% Parte 1 - análise inicial dos dados
ecg_data = sio.loadmat('signal1.mat')
ecg_signal = np.array(ecg_data['signal1'][0]).T

Fs = 200
Ts = 1/Fs
ecg_time = np.arange(start = 0, stop = (Ts*ecg_signal.shape[0]), step = Ts)

plt.figure(figsize=(15,6))
plt.plot(ecg_time, ecg_signal,
        label='Dado do ECG',
        linestyle = 'solid',
        linewidth = 1)
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('Sinal do ECG em função do tempo')

mag_ecg_signal, f = mag_of_FFT(ecg_signal, Fs)

plt.figure(figsize = (15,6))
plt.plot(f, mag_ecg_signal,
        label='Magnitude da FFT do ECG',
        linestyle = 'solid',
        marker = 'o',
        linewidth = 1)
plt.ylabel("Magnitude")
plt.xlabel("Frequencia")
plt.title("Magnitude da FFT")

# %% Parte 2: filtragem do sinal com filtro média-móvel

plt.figure(3, figsize=(15,6))
plt.plot(ecg_time, ecg_signal,
        label='Dado do ECG',
        linestyle = 'solid',
        linewidth = 1)

plt.figure(4, figsize=(15,6))
plt.plot(f, mag_ecg_signal,
        label='Magnitude da FFT do ECG',
        linestyle = 'solid',
        marker = 'o',
        linewidth = 1)

for A in [3, 5, 7, 9]:
    filtro = 1/A*np.ones(A)
    ecg_signal_c = np.convolve(ecg_signal, filtro, mode = "full")
    # precisamos aumentar o vetor de tempo porque o vetor ecg_signal_c é M+N+1
    ecg_time_c = np.arange(start = 0, stop = (Ts*ecg_signal_c.shape[0]), step = Ts)

    plt.figure(3, figsize=(15,6))
    plt.plot(ecg_time_c, ecg_signal_c,
            label='Média de ' + str(A) + " amostras",
            linestyle = 'solid',
            linewidth = 1)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude')
    plt.title('Sinal do ECG em função do tempo')

    mag_ecg_signal_c, f_c = mag_of_FFT(ecg_signal_c, Fs)

    plt.figure(4, figsize=(15,6))
    plt.plot(f_c, mag_ecg_signal_c,
            label='FFT filtrada por ' + str(A) + " amostras",
            linestyle = 'solid',
            marker = 'o',
            linewidth = 1)
    plt.ylabel("Magnitude")
    plt.xlabel("Frequencia")
    plt.title("Magnitude da FFT")


for i in range(1,5):
      plt.figure(i)
      plt.grid()
      plt.legend()
      plt.tight_layout()
      plt.savefig(fname = "fig " + str(i) + ".jpeg", dpi = 600)
