# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:27:30 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de Sinais Biológicos
Aula 4: Janelamento
Exercício 3: Janela de Hamming
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import read

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

# %% Parte 1 - análise inicial dos dados con janela retangular
Fs, baleia_data = read('whale.wav')

baleia_signal = np.array(baleia_data[9000:9000+4096], dtype=np.float64 ) - baleia_data.mean()

Ts = 1/Fs
baleia_time = np.arange(start = 0, stop = (Ts*baleia_signal.shape[0]), step = Ts)

plt.figure(figsize=(15,6))
plt.plot(baleia_time, baleia_signal,
        label='Dado do canto da baleia',
        linestyle = 'solid',
        linewidth = 1)
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('Sinal do canto da baleia em função do tempo')

mag_baleia, f = mag_of_FFT(baleia_signal, Fs)

plt.figure(figsize = (15,6))
plt.plot(f, mag_baleia,
        label='Magnitude da FFT do canto da baleia',
        linestyle = 'solid',
        marker = 'o',
        linewidth = 1)
plt.ylabel("Magnitude")
plt.xlabel("Frequencia [Hz]")
plt.title("Magnitude da FFT")

# %% Parte 2: aquisição dos dados pela janela de Hamming

filtro = np.hamming(baleia_signal.shape[0])
baleia_hamming = np.multiply(baleia_signal,filtro)
# precisamos aumentar o vetor de tempo porque o vetor ecg_signal_c é M+N+1
baleia_time_hamming = np.arange(start = 0, stop = (Ts*baleia_hamming.shape[0]), step = Ts)

plt.figure(3, figsize=(15,6))
plt.plot(baleia_time, baleia_signal,
        label='Janela retangular',
        linestyle = 'solid',
        linewidth = 1)

plt.plot(baleia_time_hamming, baleia_hamming,
        label='Janela de Hamming',
        linestyle = 'solid',
        linewidth = 1)
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('Sinal do canto da baleia em função do tempo')

mag_baleia_h, f_c = mag_of_FFT(baleia_hamming, Fs)

plt.figure(figsize = (15,6))
plt.plot(f, 20*np.log10(mag_baleia),
        label='Janela retangular',
        linestyle = 'solid',
        # marker = 'o',
        linewidth = 1)
plt.plot(f_c, 20*np.log10(mag_baleia_h),
        label='Janela de Hamming',
        linestyle = 'solid',
        # marker = 'o',
        linewidth = 1)
# plt.yscale("log")
plt.ylabel("Magnitude [db]")
plt.xlabel("Frequencia")
plt.title("Magnitude da FFT")

plt.figure(figsize = (15,6))
plt.plot(f, mag_baleia,
        label='Janela retangular',
        linestyle = 'solid',
        # marker = 'o',
        linewidth = 1)
plt.plot(f_c, mag_baleia_h,
        label='Janela de Hamming',
        linestyle = 'solid',
        # marker = 'o',
        linewidth = 1)
plt.ylabel("Magnitude")
plt.xlabel("Frequencia")
plt.title("Magnitude da FFT")


for i in range(1,6):
      plt.figure(i)
      plt.grid()
      plt.legend()
      plt.tight_layout()
      plt.savefig(fname = "fig " + str(i) + ".jpeg", dpi = 600)
