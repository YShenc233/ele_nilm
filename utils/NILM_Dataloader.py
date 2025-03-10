import torch.utils.data as data_utils
import torch  
from utils.dataset import BalancedNILMDataset

class NILMDataloader:
    """
    DataLoader for NILM datasets.
    Creates PyTorch DataLoader objects for training and validation.
    """
    
    def __init__(self, args, ds_parser, pretrain=False, balance_dataset=False):
        """
        Initialize the NILM data loader.
        
        Args:
            args: Arguments containing batch size and other parameters
            ds_parser: Data processor object containing dataset methods
            pretrain: Boolean indicating whether to use pretraining datasets
            balance_dataset: Whether to use balanced dataset to handle class imbalance
        """
        self.args = args
        self.mask_prob = args.mask_prob
        self.batch_size = args.batch_size
        self.balance_dataset = balance_dataset

        # Get appropriate datasets based on whether we're pretraining or not
        if pretrain:
            self.train_dataset, self.val_dataset = ds_parser.get_pretrain_datasets(mask_prob=self.mask_prob)
        else:
            train, val = ds_parser.get_train_datasets()
            
            # Use balanced training dataset if specified
            if balance_dataset:
                val_end = int(ds_parser.val_size * len(ds_parser.x))
                x_train = ds_parser.x[val_end:]
                y_train = ds_parser.y[val_end:]
                status_train = ds_parser.status[val_end:]
                
                self.train_dataset = BalancedNILMDataset(
                    x_train, y_train, status_train,
                    ds_parser.window_size, ds_parser.window_stride,
                    min_positive_ratio=0.2  # Adjust this based on data characteristics
                )
                self.val_dataset = val
            else:
                self.train_dataset, self.val_dataset = train, val

    def get_dataloaders(self):
        """
        Creates and returns DataLoader objects for training and validation.
        
        Returns:
            train_loader: DataLoader for training dataset
            val_loader: DataLoader for validation dataset
        """
        train_loader = self._get_loader(self.train_dataset, shuffle=True)
        val_loader = self._get_loader(self.val_dataset, shuffle=False)
        return train_loader, val_loader

    def _get_loader(self, dataset, shuffle=True):
        """
        Creates a DataLoader for a given dataset.
        
        Args:
            dataset: PyTorch Dataset object
            shuffle: Whether to shuffle the data
            
        Returns:
            dataloader: PyTorch DataLoader object
        """
        # Check for empty dataset
        if len(dataset) == 0:
            print("Warning: Dataset is empty or too small for the window size.")
            # Return empty dataloader
            return data_utils.DataLoader(dataset, batch_size=1)
        
        dataloader = data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4 if torch.cuda.is_available() else 0,
            drop_last=False
        )
        return dataloader