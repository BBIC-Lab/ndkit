import numpy as np
from sklearn.model_selection import train_test_split


def channel_norm(data):
    """
    Normalize each channel using mean and std computed over (N, T).
    
    Args:
        data (np.ndarray): Array of shape (N, T, D)
    
    Returns:
        np.ndarray: Normalized data with the same shape.
    """
    assert isinstance(data, np.ndarray) and data.ndim == 3, "Input must be 3D (N, T, D)."

    mean = np.mean(data, axis=(0, 1), keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)

    return (data - mean) / (std + 1e-8)   # avoid division by zero


def downsample(data, rate=10, mode="mean"):
    """
    Downsample along the time dimension.
    
    Args:
        data (np.ndarray): Input of shape (N, T, D)
        rate (int): Downsampling factor.
        mode (str): One of {"mean", "sum", "first", "last"}.
    
    Returns:
        np.ndarray: Downsampled array of shape (N, T // rate, D)
    """
    assert isinstance(data, np.ndarray) and data.ndim == 3, "Input must be 3D (N, T, D)."

    N, T, D = data.shape
    new_T = T // rate

    # Trim to ensure divisibility, then reshape into blocks
    trimmed = data[:, :new_T * rate, :].reshape(N, new_T, rate, D)

    if mode == "mean":
        return trimmed.mean(axis=2)
    elif mode == "sum":
        return trimmed.sum(axis=2)
    elif mode == "first":
        return trimmed[:, :, 0]
    elif mode == "last":
        return trimmed[:, :, -1]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from ['mean', 'sum', 'first', 'last'].")
    

def partition(condition, test_frac):

    """
    Partition trials into training and testing sets.

    Inputs
    ------
    condition: list (1 x number of trials) of condition IDs
        If there is no condition structure, all entries will be NaNs.

    test_frac: fraction of trials in the data set that should be reserved for testing

    Outputs
    -------
    train_idx: 1D numpy array of indices for trials that should be used for training

    test_idx: 1D numpy array of indices for trials that should be used for testing
   
    """

    # Number the trials.
    trial_idx = np.arange(len(condition))
    
    # Partition the data differently depending on whether there is condition structure.
    if not np.all(np.isnan(condition)):

        # Try to maintain equal representation from different conditions in the train
        # and test sets. If the test set is so small that it can't sample at least
        # one from each condition, then don't bother trying to stratify the split.
        n_conds = len(np.unique(np.array(condition)))
        n_test = int(test_frac*len(condition))
        if n_test >= n_conds:
            train_idx, test_idx = train_test_split(trial_idx, test_size=test_frac, stratify=condition, random_state=42)
        else:
            train_idx, test_idx = train_test_split(trial_idx, test_size=test_frac, random_state=42)
    else:

        # Divide the trials up randomly into train and test sets.
        train_idx, test_idx = train_test_split(trial_idx, test_size=test_frac, random_state=42)
    return train_idx, test_idx

def bin_spikes(spikes, bin_size):

    """
    Bin spikes in time.

    Inputs
    ------
    spikes: numpy array of spikes (neurons x time)

    bin_size: number of time points to pool into a time bin

    Outputs
    -------
    S: numpy array of spike counts (neurons x bins)
   
    """

    # Get some useful constants.
    [N, n_time_samples] = spikes.shape
    K = int(n_time_samples/bin_size) # number of time bins

    # Count spikes in bins.
    S = np.empty([N, K])
    for k in range(K):
        S[:, k] = np.sum(spikes[:, k*bin_size:(k+1)*bin_size], axis=1)

    return S

def append_history(S, tau_prime):

    """
    Augment spike count array with additional dimension for recent spiking history.

    Inputs
    ------
    S: numpy array of spike counts (neurons x bins)

    tau_prime: number of historical time bins to add (not including current bin)

    Outputs
    -------
    S_aug: tensor of spike counts (neurons x bins x recent bins)
   
    """

    # Get some useful constants.
    [N, K] = S.shape # [number of neurons, number of bins]

    # Augment matrix with recent history.
    S_aug = np.empty([N, K, tau_prime+1])
    for i in range(-tau_prime,0): 
        S_aug[:, :, i+tau_prime] = np.hstack((np.full([N,-i], np.nan), S[:, :i]))
    S_aug[:, :, tau_prime] = S 

    return S_aug

def seq_process(Delta, tau_prime, neural, behavior):
    neural = [bin_spikes(sp, Delta) for sp in neural]

    # Reformat observations to include recent history.
    neural = [append_history(s, tau_prime) for s in neural]

    # Downsample kinematics to bin width.
    behavior = [z[:,Delta-1::Delta] for z in behavior]

    # Remove samples on each trial for which sufficient spiking history doesn't exist.
    neural = [x[:,tau_prime:,:] for x in neural]
    behavior = [z[:,tau_prime:] for z in behavior]

    # Concatenate X and Z across trials (in time bin dimension) and rearrange dimensions.
    neural = np.moveaxis(np.concatenate(neural,axis=1), [0, 1, 2], [2, 0, 1])
    behavior = np.concatenate(behavior, axis=1).T
    return neural, behavior