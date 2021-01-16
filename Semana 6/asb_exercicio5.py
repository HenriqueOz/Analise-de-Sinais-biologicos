# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:19:02 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de Sinais Biológicos
Aula 5: Filtros 2
Exercício 3: Filtro FIR
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

def firDesign(order, Fc, Fs):
    Fcn = Fc/Fs             #Normalized cut frequency
    n = 1024                #number of points
    posFc = int(n * Fcn); #cut frequency point (point position)
    H = np.concatenate(([np.ones(posFc)],[np.zeros(n - 2*posFc)],[np.ones(posFc)]), axis = 1)
    hm = np.fft.ifft(H);           #IFFT (h mirrored)
    hs = np.fft.ifftshift(hm);      #swaps the left and right halves of the IFFT
    h = hs[0,int(n/2-(order/2)):int(n/2 + order/2)].real
    f = np.arange(start = 0, stop = Fs/2, step = Fs/n)
    return h, H, hs, f

# %% Parte 1 - análise inicial dos dados com janela retangular
signal2_data = sio.loadmat('signal2.mat')
signal = np.array(signal2_data['signal2'][:,0]).T
signal = signal - signal.mean()
Fs = signal2_data['Fs2']

Ts = 1/Fs
time = np.arange(start = 0, stop = (Ts*signal.shape[0]), step = Ts)

plt.figure(figsize=(15,6))
plt.plot(time, signal,
        label='Dado de signal2.mat',
        linestyle = 'solid',
        linewidth = 1)
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('Sinal presente em signal2')

mag_signal, f = mag_of_FFT(signal, Fs)

plt.figure(figsize = (15,6))
plt.plot(f, mag_signal,
        label='Magnitude da FFT de signal2.mat',
        linestyle = 'solid',
        marker = 'o',
        linewidth = 1)
plt.ylabel("Magnitude")
plt.xlabel("Frequencia [Hz]")
plt.title("Magnitude da FFT")
# %% Parte 2: Design do FIR passa-baixa
janela = np.hamming(signal.shape[0])
signal_h = np.multiply(signal,janela)
time_h = np.arange(start = 0, stop = (Ts*signal_h.shape[0]), step = Ts)
mag_signal_h, f_h = mag_of_FFT(signal_h, Fs)

plt.figure(5, figsize=(15,6))
plt.plot(time, signal,
        label='Janela retangular',
        linestyle = 'solid',
        linewidth = 1)
plt.plot(time_h, signal_h,
        label='Janela de Hamming',
        linestyle = 'solid',
        linewidth = 1)

plt.figure(6, figsize = (15,6))
plt.plot(f, 20*np.log10(mag_signal),
        label='Janela retangular',
        linestyle = 'solid',
        # marker = 'o',
        linewidth = 1)
plt.plot(f_h, 20*np.log10(mag_signal_h),
        label='Janela de Hamming',
        linestyle = 'solid',
        # marker = 'o',
        linewidth = 1)

plt.figure(7, figsize = (15,6))
plt.plot(f, mag_signal,
        label='Janela retangular',
        linestyle = 'solid',
        # marker = 'o',
        linewidth = 1)
plt.plot(f_h, mag_signal_h,
        label='Janela de Hamming',
        linestyle = 'solid',
        # marker = 'o',
        linewidth = 1)

for order in [3,25,49,131]:
    Fc = 50
    filtro, filtro_FFT, filtro_tempo, f_filtro = firDesign(order, Fc, Fs)

    plt.figure(3, figsize=(15,6))
    plt.plot(f_filtro,filtro_FFT[0,0:int(filtro_FFT.shape[1]/2)],
            label='Ordem ' + str(order),
            linestyle = 'solid',
            marker = 'o',
            linewidth = 1)
    plt.xlabel('Frequência [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Filtro passa baixa com frequência de corte de 50 Hz')

    plt.figure(4, figsize=(15,6))
    plt.plot(filtro_tempo[0,:],
            label='Ordem ' + str(order),
            linestyle = 'solid',
            marker = 'o',
            linewidth = 1)
    plt.xlabel('Amostras')
    plt.ylabel('Amplitude')
    plt.title('Forma do filtro no tempo')

    plt.figure(8, figsize=(15,6))
    plt.plot(filtro, label='Ordem ' + str(order))
    plt.xlabel('Amostras')
    plt.ylabel('Amplitude')
    plt.title('Formato do filtro')

    # %% Parte 3: aquisição dos dados pela janela de Hamming e filtragem pelo FIR passa-baixa

    signal_f = np.convolve(signal, filtro, mode = "full")
    janela = np.hamming(signal_f.shape[0])
    signal_f = np.multiply(signal_f,janela)

    # precisamos aumentar o vetor de tempo porque o vetor ecg_signal_c é M+N+1
    time_f = np.arange(start = 0, stop = (Ts*signal_f.shape[0]), step = Ts)
    mag_signal_f, f_f = mag_of_FFT(signal_f, Fs)

    plt.figure(5, figsize=(15,6))
    plt.plot(time_f, signal_f,
            label='Ordem ' + str(order),
            linestyle = 'solid',
            linewidth = 1)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude')
    plt.title('Sinal do canto da baleia em função do tempo')

    plt.figure(6, figsize = (15,6))
    plt.plot(f_f, 20*np.log10(mag_signal_f),
            label='Ordem ' + str(order),
            linestyle = 'solid',
            # marker = 'o',
            linewidth = 1)
    # plt.xscale("log")
    plt.ylabel("Magnitude [db]")
    plt.xlabel("Frequencia")
    plt.title("Magnitude da FFT")

    plt.figure(7, figsize = (15,6))
    plt.plot(f_f, mag_signal_f,
            label='Ordem ' + str(order),
            linestyle = 'solid',
            # marker = 'o',
            linewidth = 1)
    # plt.yscale("log")
    plt.ylabel("Magnitude")
    plt.xlabel("Frequencia")
    plt.title("Magnitude da FFT")

for i in range(1,9):
      plt.figure(i)
      plt.grid()
      plt.legend()
      plt.tight_layout()
      plt.savefig(fname = "fig " + str(i) + ".jpeg", dpi = 600)
