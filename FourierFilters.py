"""
@author Maria Eduarda Kawakami (https://github.com/madukm)
@date 27 October, 2020
@brief Introduction to DFT and IDFT.
@details
Implementation of Discrete Fourier Transform in 1D and 2D and Inverse DFT.
In Main function, I apply these functions and do some comparisons.
But you can feel free to use only the other functions.
"""

import numpy as np
import imageio

def DFT1D(f):
    F = np.zeros(f.shape, dtype=np.complex64)
    n = f.shape[0]

    x =  np.arange(n)
    for u in np.arange(n):
        F[u] = np.sum(f*np.exp((1j*2*np.pi* u*x) / n))

    return F/np.sqrt(n)

def DFT2D(f):
    F = np.zeros(f.shape, dtype=np.complex64)
    n, m = f.shape[0:2]

    x = np.arange(n).reshape(n,1)
    y = np.arange(m).reshape(1,m)

    for u in np.arange(n):
        for v in np.arange(m):
            F[u,v] = np.sum(f * np.exp( (-1j*2*np.pi) * (((u*x)/n)+((v*y)/m))))

    return F/np.sqrt(n*m)

def IDFT2D(F):
    f = np.zeros(F.shape, dtype=np.complex64)
    n, m = F.shape[0:2]

    u = np.arange(n).reshape(n,1)
    v = np.arange(m).reshape(1,m)

    for x in np.arange(n):
        for y in np.arange(m):
            f[x,y] = np.sum(F * np.exp( (1j*2*np.pi) * (((u*x)/n)+((v*y)/m))))

    return np.abs(np.real(f/np.sqrt(n*m)))

def fourierSpectrum(F):
    return np.abs(F)

#Root Mean Square Error
def RMSE(f, g):
    n, m = f.shape()
    error = np.subtract(input_img.astype(np.float), output_img.astype(np.float))
    error = np.divide(np.sum(np.square(error), n*m))
    return np.round(np.sqrt(error), 4)

"""
In the main, we are applying DFT and finding the peak of the Fourier Spectrum
Then setting to zero all coefficients below the second peak and performing the IDFT.
Also, we compute the mean of the original image and after the processing.
"""
def main():
    filename = str(input()).rstrip()
    input_img = imageio.imread(filename)
    T = float(input()) #threshold

    # Apply DFT 
    F1 = DFT2D(input_img)
    # Get the Fourier Spectrum (|F(u,v)|)
    spectrum = fourierSpectrum(F1)
    # Second peak
    p2 = np.partition(spectrum.flatten(), -2)[-2]

    # New threshold (p2 * T)
    print("Threshold=%.4f" % (p2*T))

    # Set to zero all coefficients under the new threshold
    F1[np.abs(F1)<p2*T] = 0

    #Counting how many coefficients are under this condition
    coefficients = np.count_nonzero(np.abs(F1)<p2*T)
    print("Filtered Coefficients=%d" % coefficients)

    # Apply IDFT
    filtered = IDFT2D(F1)

    originalMean = np.mean(input_img)
    newMean = np.mean(filtered)
    print("Original Mean=%.2f" % originalMean)
    print("New Mean=%.2f" % newMean)

if __name__ == '__main__':
    main()
