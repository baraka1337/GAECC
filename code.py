from code import BER, bin_to_sign, generate_parity_check_matrix
import numpy as np

import pyldpc
from main import DEFAULT_BP_MAX_ITER


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

    snr = EbN0 + 10.0 * np.log10(2 * rate)
    return np.sqrt(1.0 / (10.0 ** (snr / 10.0)))


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
    snr = EbN0 + 10.0 * np.log10(2 * rate)
    return np.sqrt(1.0 / (10.0 ** (snr / 10.0)))


def EbN0_to_snr(EbN0, rate):
    """
    Converts Eb/N0 (Energy per Bit to Noise Power Spectral Density ratio) to SNR (Signal-to-Noise Ratio).

    Parameters:
    - EbN0 (float): The Eb/N0 ratio in dB.
    - rate (float): The data rate in bits per second.

    Returns:
    - snr (float): The SNR ratio in dB.
    """
    return EbN0 + 10.0 * np.log10(2 * rate)


def snr_to_EbN0(snr, rate):
    """
    Convert Signal-to-Noise Ratio (SNR) to Energy per Bit to Noise Power Spectral Density (Eb/N0).

    Parameters:
    - snr (float): Signal-to-Noise Ratio in decibels (dB).
    - rate (float): Data rate in bits per second (bps).

    Returns:
    - float: Energy per Bit to Noise Power Spectral Density (Eb/N0) in decibels (dB).
    """
    return snr - 10.0 * np.log10(2 * rate)


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


def G_from_solution(solution, k):
    """
    Constructs the G matrix from a given solution and dimension k.

    Parameters:
    solution (ndarray): The solution array.
    k (int): The dimension of the identity matrix.

    Returns:
    ndarray: The G matrix constructed by horizontally stacking the identity matrix with the solution array.
    """
    return np.hstack((np.eye(k), solution))


def AWGN_channel(x, sigma):
    """
    Applies Additive White Gaussian Noise (AWGN) to the input signal.

    Parameters:
    - x: numpy array, input signal
    - sigma: float, standard deviation of the Gaussian noise

    Returns:
    - y: numpy array, signal with AWGN applied
    """
    mean = 0
    z = np.random.normal(mean, sigma, x.shape)
    y = bin_to_sign(x) + z
    return y


def test_G(G, sample_size, sigma, snr, H=None, bp_iter=DEFAULT_BP_MAX_ITER):
    """
    Test the performance of a given generator matrix G.

    Args:
        G (ndarray): The generator matrix.
        sample_size (int): The number of samples to test.
        sigma (float): The standard deviation of the AWGN channel.
        snr (float): The signal-to-noise ratio.
        H (ndarray, optional): The parity check matrix. If not provided, it will be generated.
        bp_iter (int, optional): The maximum number of iterations for belief propagation decoding.

    Returns:
        float: The bit error rate (BER) of the decoded samples.
    """
    if H is None:
        H = generate_parity_check_matrix(G)
    x_vec = np.zeros((sample_size, G.shape[1]))
    y_vec = AWGN_channel(x_vec, sigma)

    x_pred_vec = pyldpc.decode(H, y_vec.T, snr, bp_iter)
    x_pred_vec = x_pred_vec.T
    return BER(x_vec, x_pred_vec)


# type: ignore
