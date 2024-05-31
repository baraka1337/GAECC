import numpy as np



def bin_to_sign(x):
    """
    Converts a binary value to its signed representation.

    Parameters:
    x (int): The binary value to be converted.

    Returns:
    int: The signed representation of the binary value.
    """
    return 1 - 2 * x


def EbN0_to_std(EbN0, rate):
    """
    Convert Eb/N0 to standard deviation of the noise.

    Parameters:
    - EbN0 (float): The ratio of energy per bit to noise power spectral density.
    - rate (float): The data rate in bits per second.

    Returns:
    - std (float): The standard deviation of the noise.
    """
    
    snr = EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))


def BER(x_pred, x_gt):
    """
    Calculates the Bit Error Rate (BER) between two binary sequences.

    Parameters:
    x_pred (array-like): The predicted binary sequence.
    x_gt (array-like): The ground truth binary sequence.

    Returns:
    float: The calculated Bit Error Rate (BER).

    """
    return np.mean((x_pred != x_gt)).item()


def EbN0_to_std(EbN0, rate):
    """
    Convert Eb/N0 to standard deviation of the noise.

    Parameters:
    - EbN0 (float): The ratio of energy per bit to noise power spectral density.
    - rate (float): The data rate in bits per second.

    Returns:
    - std (float): The standard deviation of the noise.

    """
    snr = EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))


def EbN0_to_snr(EbN0, rate):
    """
    Converts Eb/N0 (Energy per Bit to Noise Power Spectral Density ratio) to SNR (Signal-to-Noise Ratio).

    Parameters:
    - EbN0 (float): The Eb/N0 ratio in dB.
    - rate (float): The data rate in bits per second.

    Returns:
    - snr (float): The SNR ratio in dB.
    """
    return EbN0 + 10. * np.log10(2 * rate)

def snr_to_EbN0(snr, rate):
    """
    Convert Signal-to-Noise Ratio (SNR) to Energy per Bit to Noise Power Spectral Density (Eb/N0).

    Parameters:
    - snr (float): Signal-to-Noise Ratio in decibels (dB).
    - rate (float): Data rate in bits per second (bps).

    Returns:
    - float: Energy per Bit to Noise Power Spectral Density (Eb/N0) in decibels (dB).
    """
    return snr - 10. * np.log10(2 * rate)

def generate_parity_check_matrix(G):
    """
    Generates the parity check matrix (H) from the generator matrix (G).

    Parameters:
    G (numpy.ndarray): The generator matrix.

    Returns:
    numpy.ndarray: The parity check matrix.

    """
    k, n = G.shape
    H = np.hstack((G[:, k:].T.astype(int) ^ 1, np.eye(n - k)))
    return H
 # type: ignore
