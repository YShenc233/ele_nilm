import torch
import torch.utils.data as data_utils
import random
import numpy as np

class NILMDataset(data_utils.Dataset):
    """
    Dataset for Non-Intrusive Load Monitoring (NILM).
    Provides data windows of aggregate power, appliance power, and ON/OFF status.
    """
    
    def __init__(self, x, y, status, window_size=480, stride=30):
        """
        Initialize NILM dataset.
        
        Args:
            x: Aggregate power data
            y: Appliance power data
            status: ON/OFF status data
            window_size: Size of each data window
            stride: Stride between consecutive windows
        """
        self.x = x
        self.y = y
        self.status = status
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
    # Original code which can return negative values if window_size > len(self.x)
    # return int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)
    
    # Fixed version: return 0 for empty datasets instead of negative values
        if len(self.x) <= self.window_size:
            return 0  # Return 0 if dataset doesn't have enough data for one window
        else:
            return int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)

    def __getitem__(self, index):
        """Get a specific window of data"""
        start_index = index * self.stride
        end_index = np.min((len(self.x), index * self.stride + self.window_size))
        
        x = self.padding_seqs(self.x[start_index: end_index])
        y = self.padding_seqs(self.y[start_index: end_index])
        status = self.padding_seqs(self.status[start_index: end_index])
        
        x = torch.tensor(x).view((1, -1))
        y = torch.tensor(y).view((1, -1))
        status = torch.tensor(status).view((1, -1))
        
        return x, y, status

    def padding_seqs(self, in_array):
        """Pad sequences to window_size"""
        if len(in_array) == self.window_size:
            return in_array
        try:
            out_array = np.zeros((self.window_size, in_array.shape[1]))
        except:
            out_array = np.zeros(self.window_size)

        out_array[:len(in_array)] = in_array
        return out_array


class BalancedNILMDataset(NILMDataset):
    """
    Balanced dataset for NILM that oversamples windows with ON states.
    Helps with class imbalance where OFF states are much more common.
    """
    
    def __init__(self, x, y, status, window_size=480, stride=30, min_positive_ratio=0.3):
        """
        Initialize balanced NILM dataset.
        
        Args:
            x: Aggregate power data
            y: Appliance power data
            status: ON/OFF status data
            window_size: Size of each data window
            stride: Stride between consecutive windows
            min_positive_ratio: Minimum ratio of ON states to consider a window as positive
        """
        super().__init__(x, y, status, window_size, stride)
        self.min_positive_ratio = min_positive_ratio
        self._find_positive_samples()
        
    def _find_positive_samples(self):
        """Find windows with significant ON states"""
        self.positive_indices = []
        total_samples = self.__len__()
        
        for i in range(total_samples):
            start_index = i * self.stride
            end_index = np.min((len(self.x), i * self.stride + self.window_size))
            status_sample = self.status[start_index:end_index]
            
            if len(status_sample) > 0:
                positive_ratio = np.mean(status_sample)
                if positive_ratio >= self.min_positive_ratio:
                    self.positive_indices.append(i)
        
        print(f"Found {len(self.positive_indices)} samples with ON ratio >= {self.min_positive_ratio}")
    
    def __getitem__(self, index):
        """Get a window, with higher probability of selecting windows with ON states"""
        # With 50% probability, sample a window with significant ON states
        if random.random() < 0.5 and len(self.positive_indices) > 0:
            index = random.choice(self.positive_indices)
        
        # Call parent class method to get data
        return super().__getitem__(index)


class Pretrain_Dataset(NILMDataset):
    """
    Dataset for pretraining NILM models with masked input.
    Masks random parts of the input sequence to help model learn robust features.
    """
    
    def __init__(self, x, y, status, window_size=480, stride=30, mask_prob=0.25):
        """
        Initialize pretraining dataset.
        
        Args:
            x: Aggregate power data
            y: Appliance power data
            status: ON/OFF status data
            window_size: Size of each data window
            stride: Stride between consecutive windows
            mask_prob: Probability of masking each time step
        """
        self.x = x
        self.y = y
        self.status = status
        self.window_size = window_size
        self.stride = stride
        self.mask_prob = mask_prob

    def __getitem__(self, index):
        """Get a window with random masking applied"""
        start_index = index * self.stride
        end_index = np.min((len(self.x), index * self.stride + self.window_size))
       
        x = self.padding_seqs(self.x[start_index: end_index]).copy()
        y = self.padding_seqs(self.y[start_index: end_index]).copy()
        status = self.padding_seqs(self.status[start_index: end_index]).copy()

        # Apply masking
        for i in range(len(x)):
            prob = random.random()
            if prob <= self.mask_prob:
                # For masked positions: 80% set to -1, 10% to random noise, 10% unchanged
                prob = random.random()
                x[i] = -1 if prob < 0.8 else np.random.normal() if prob < 0.9 else x[i]
            else:
                # For unmasked positions, we don't try to predict y and status
                y[i] = -1
                status[i] = -1

        x = torch.tensor(x).view((1, -1))
        y = torch.tensor(y).view((1, -1))
        status = torch.tensor(status).view((1, -1))
        
        return x, y, status