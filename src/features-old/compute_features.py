"""
Calculate the model features. This code is intended to be an implementation of
the feature set described in Tsiouris et al. (2018); unless explicitly noted as
such, all deviations from the description there are bugs.
"""

import numpy as np
import scipy.stats
import pywt
import networkx

from bct.algorithms.distance import efficiency_wei
from networkx.algorithms.cluster import clustering
from networkx.linalg.graphmatrix import adjacency_matrix
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.shortest_paths.generic import shortest_path_length, \
average_shortest_path_length

SAMPLE_RATE = 256

def zero_crossings(signal):
    """
    Count the number of times that the signal crosses the zero voltage point
    """
    return len(np.where(np.diff(np.signbit(signal)))[0])

def peak_to_peak(signal):
    """
    Calculate the range of the signal (how much the voltage varies absolutely)
    """
    return np.max(signal) - np.min(signal)

def total_signal_area(signal):
    """
    Compute the total area under the signal.

    Note here that this computes unsigned area, and so cannot just be replaced
    with the trapezoidal method. This is actually just an extension of the
    trapezoidal method to take care of the case where the function crosses the
    x axis. It's admittedly a bit tricky to follow; this is loosely adapted from
    from the output of Mathematica evaluating the integral of an absolute value
    on a small interval.
    """
    c = signal[:-1]
    d = signal[1:]
    small = np.finfo(signal.dtype).tiny
    sc = np.sign(c + small)
    sd = np.sign(d)
    dx = 1/SAMPLE_RATE
    parts = sc * (c ** 2 - sc * sd * d ** 2) / (2 * (c - d))
    nans = np.isnan(parts)
    parts[nans] = np.abs(c[nans])
    return np.sum(parts) * dx

def total_signal_energy(signal):
    """
    Calculate the signal's overall energy; don't bother calculating the subparts
    here.
    """
    return np.sum(signal ** 2)

def band_energy(fft, freqs, low, high):
    """
    Calculate the energy in a particular frequency band, given a Fourier
    transformed set `fft` with accompanying frequencies `freqs`, between `low`
    and `high` (measured in hertz)
    """
    selector = np.logical_and(freqs > low, freqs < high)
    return np.trapz(fft[selector], freqs[selector], dx=1/SAMPLE_RATE)

def energy_percentages(signal):
    """
    Calculate the percent of energy in various frequency bands. Those bands are:

    * Delta (<= 3 Hz)
    * Theta (4-7 Hz)
    * Alpha (8-13 Hz)
    * Beta (14-30 Hz)
    * Lower Gamma (30-55 Hz)
    * Higher Gamma (65-110 Hz)
    """
    padded = np.array([0] * 100 + list(signal) + [0] * 100)
    freqs = np.fft.rfftfreq(len(padded)) * SAMPLE_RATE
    fft = np.abs(np.fft.rfft(padded))
    overall = np.trapz(fft, freqs)
    delta = band_energy(fft, freqs, 0, 3)
    theta = band_energy(fft, freqs, 4, 7)
    alpha = band_energy(fft, freqs, 8, 13)
    beta = band_energy(fft, freqs, 14, 30)
    gamma1 = band_energy(fft, freqs, 30, 55)
    gamma2 = band_energy(fft, freqs, 65, 110)
    return np.array([delta, theta, alpha, beta, gamma1, gamma2]) / overall

def discrete_wavelet(signal):
    """
    Calculate the details coefficients of the discrete wavelet transform. We
    discard the approximation coefficients because they're mostly noise. Note
    that this just averages the coefficients from each level; this was the most
    reasonable way I could think of to gather a single representative value.
    This is essentially making the assumption that there are no important
    variations in the signal within each five-second segment, which seems
    reasonable.
    """
    coeffs = pywt.wavedec(signal, 'db4', level=7)
    return [coeff.mean() for coeff in coeffs[1:]]

def autocorr(x, t=1):
    """
    Calculate signal autocorrelation with a specified time offset. Here it's one
    sample by default.

    This code taken from https://stackoverflow.com/questions/643699
    """
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0, 1]

def decorrelation_times(signal):
    """
    Calculate the decorrelation time of a signal using the simplest, iterative
    approach: checking gradually increasing offsets and seeing how long it takes
    to follow below the critical threshold of exp(-1).
    """
    t = 0
    while autocorr(signal, t) > np.exp(-1):
        t += 1
    return t

# Graph theoretic
def make_cross_correlation_graph(signals):
    """
    Make a NetworkX graph with the connection structure defined by the
    correlations between

    Nodes are labeled by integers, not by the 10-20 codes used in the files.
    However, since nodes are always fed in with a consistent order, and because
    the consuming code sorts these, they should always come out with a
    constistent order (documented elsewhere in this repository)
    """
    G = networkx.Graph()
    for (channel_num_1, signal1) in enumerate(signals):
        for (channel_num_2, signal2) in enumerate(signals):
            if channel_num_1 < channel_num_2:
                # absolute value is necessary to prevent negative weights,
                # which screw up some of the graph metrics
                corr = abs(np.correlate(signal1, signal2).max())
                G.add_edge(channel_num_1, channel_num_2, weight=corr)
    return G

def graph_stats(G):
    """
    Compute all the graph-related statistics in the features.

    Note that since the graph is always fully connected, all of these are the
    weighted versions. For this reason, many of these functions use the
    implementations in bctpy rather than NetworkX.
    """
    # Local measures
    clustering_dict = clustering(G, weight='weight')
    adjacency = np.array(adjacency_matrix(G).todense())
    betweenness_centrality_dict = betweenness_centrality(G, weight='weight')
    paths = shortest_path_length(G, weight='weight')
    eccentricities = [max(dists.values()) for (source, dists) in sorted(paths)]
    local_measures = np.concatenate([
        [v for (k, v) in sorted(clustering_dict.items())],
        [v for (k, v) in sorted(betweenness_centrality_dict.items())],
        eccentricities
    ])
    graph_diameter = max(eccentricities)
    graph_radius = min(eccentricities)
    aspl = average_shortest_path_length(G, weight='weight')
    global_measures = np.array([
        graph_diameter,
        graph_radius,
        aspl
    ])
    return np.concatenate([local_measures, global_measures])

def all_features(channels):
    """
    Inputs: an array of readings for each channel
    Outputs: a vector containing all the features

    If the input array has length n, the output array should have length
    (n^2 + 53n + 8)/2
    """
    means = np.array(list(map(np.mean, channels)))
    variances = np.array(list(map(np.var, channels)))
    skewnesses = np.array(list(map(scipy.stats.skew, channels)))
    kurtoses = np.array(list(map(scipy.stats.kurtosis, channels)))
    stdevs = np.array(list(map(np.std, channels)))
    zero_crosses = np.array(list(map(zero_crossings, channels)))
    peak_to_peaks = list(map(peak_to_peak, channels))
    sig_areas = list(map(total_signal_area, channels))
    sig_energies = list(map(total_signal_energy, channels))
    energy_pcts = np.array(list(map(energy_percentages, channels))).ravel()
    dwts = np.array(list(map(discrete_wavelet, channels))).ravel()
    G = make_cross_correlation_graph(channels)
    max_ccs = [d['weight'] for d in dict(G.edges()).values()]
    autocorrs = np.array(list(map(decorrelation_times, channels))).ravel()
    graph_measures = graph_stats(G)

    vector = np.concatenate([
        means, variances, skewnesses, kurtoses,
        stdevs, zero_crosses, peak_to_peaks, sig_areas,
        sig_energies, energy_pcts, dwts,
        max_ccs, autocorrs,
        graph_measures
    ])
    return vector
