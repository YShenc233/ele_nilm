import numpy as np
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
from utils.metrics import compute_status

class DataProcessor:
    def __init__(self, args):
        self.args = args
        self.dataset_code = args.dataset_code
        self.data_location = self._get_data_location()
        self.house_indices = args.house_indicies
        self.appliance_names = args.appliance_names
        self.sampling = args.sampling
        self.normalize = args.normalize
        
        self.cutoff = [args.cutoff[appl] for appl in ['aggregate' if self.dataset_code != 'refit' else 'Aggregate'] + args.appliance_names]
        self.threshold = [args.threshold[appl] for appl in args.appliance_names]
        self.min_on = [args.min_on[appl] for appl in args.appliance_names]
        self.min_off = [args.min_off[appl] for appl in args.appliance_names]

        self.val_size = args.validation_size
        self.window_size = args.window_size
        self.window_stride = args.window_stride

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)  # 获取项目根目录
        self.processed_dir = Path(project_dir) / 'data' / 'processed' / self.dataset_code / args.appliance_names[0]
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Check if processed data exists, otherwise process it
        if not self._check_processed_data_exists():
            print(f"Processing data for {self.dataset_code}...")
            self.x, self.y = self.load_and_process_data()
            self._save_processed_data()
        else:
            print(f"Loading processed data for {self.dataset_code}...")
            self.x, self.y, self.status, self.x_mean, self.x_std = self._load_processed_data()
    
    def _check_processed_data_exists(self):
        """Check if processed data files exist"""
        files = [
            self.processed_dir / "x_data.csv",
            self.processed_dir / "y_data.csv",
            self.processed_dir / "status.csv",
            self.processed_dir / "stats.csv"
        ]
        return all(f.exists() for f in files)
    
    def _save_processed_data(self):
        """Save processed data to CSV files"""
        # Calculate status
        self.status = self._compute_status(self.y)
        
        # Calculate stats
        if self.normalize == 'mean':
            self.x_mean = np.mean(self.x)
            self.x_std = np.std(self.x)
            
            # Save normalization stats
            stats_df = pd.DataFrame({'mean': [self.x_mean], 'std': [self.x_std]})
            stats_df.to_csv(self.processed_dir / "stats.csv", index=False)
            
            # Normalize data
            self.x = (self.x - self.x_mean) / self.x_std
        
        # Save processed data
        pd.DataFrame(self.x).to_csv(self.processed_dir / "x_data.csv", index=False, header=False)
        pd.DataFrame(self.y).to_csv(self.processed_dir / "y_data.csv", index=False, header=False)
        pd.DataFrame(self.status).to_csv(self.processed_dir / "status.csv", index=False, header=False)
        
        print(f"Processed data saved to {self.processed_dir}")
    
    def _load_processed_data(self):
        """Load processed data from CSV files"""
        x = pd.read_csv(self.processed_dir / "x_data.csv", header=None).values
        y = pd.read_csv(self.processed_dir / "y_data.csv", header=None).values
        status = pd.read_csv(self.processed_dir / "status.csv", header=None).values
        
        # Load stats
        stats = pd.read_csv(self.processed_dir / "stats.csv")
        x_mean = stats['mean'].values[0]
        x_std = stats['std'].values[0]
        
        return x, y, status, x_mean, x_std
    
    def _get_data_location(self):
        """Get data location based on dataset code"""
        if self.dataset_code == 'redd_lf':
            return self.args.redd_location
        elif self.dataset_code == 'uk_dale':
            return self.args.ukdale_location
        elif self.dataset_code == 'refit':
            return self.args.refit_location
        else:
            raise ValueError(f"Unknown dataset code: {self.dataset_code}")
    
    def load_and_process_data(self):
        """Load and process data based on dataset code"""
        if self.dataset_code == 'redd_lf':
            return self._load_redd_data()
        elif self.dataset_code == 'uk_dale':
            return self._load_ukdale_data()
        elif self.dataset_code == 'refit':
            return self._load_refit_data()
        else:
            raise ValueError(f"Unknown dataset code: {self.dataset_code}")
    
    def _compute_status(self, data):
        """Compute status for the data"""
        initial_status = data >= self.threshold[0]
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()

        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > self.min_off[0]]
            off_events = off_events[np.roll(off_duration, -1) > self.min_off[0]]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= self.min_on[0]]
            off_events = off_events[on_duration >= self.min_on[0]]
            assert len(on_events) == len(off_events)

        temp_status = data.copy()
        temp_status[:] = 0
        for on, off in zip(on_events, off_events):
            temp_status[on: off] = 1
        return temp_status
    
    def _load_redd_data(self):
        """Load and process REDD dataset"""
        directory = Path(self.data_location)
        entire_data = None
        
        for house_id in self.house_indices:
            if house_id not in [1, 2, 3, 4, 5, 6]:
                continue
                
            house_folder = directory.joinpath(f'house_{house_id}')
            house_label = pd.read_csv(house_folder.joinpath('labels.dat'), sep=' ', header=None)
            main_1 = pd.read_csv(house_folder.joinpath('channel_1.dat'), sep=' ', header=None)
            main_2 = pd.read_csv(house_folder.joinpath('channel_2.dat'), sep=' ', header=None)
            
            house_data = pd.merge(main_1, main_2, how='inner', on=0)
            house_data.iloc[:, 1] = house_data.iloc[:,1] + house_data.iloc[:,2]
            house_data = house_data.iloc[:, 0:2]
            
            appliance_list = house_label.iloc[:, 1].values
            app_index_dict = defaultdict(list)
            
            for appliance in self.appliance_names:
                try:
                    idx = appliance_list.tolist().index(appliance)
                    app_index_dict[appliance].append(idx+1)
                except ValueError:
                    app_index_dict[appliance].append(-1)
            
            if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                continue
            
            for appliance in self.appliance_names:
                if app_index_dict[appliance][0] == -1:
                    temp_values = house_data.copy().iloc[:, 1]
                    temp_values[:] = 0
                    temp_data = house_data.copy().iloc[:, :2]
                    temp_data.iloc[:, 1] = temp_values
                else:
                    temp_data = pd.read_csv(house_folder.joinpath(f'channel_{app_index_dict[appliance][0]}.dat'), sep=' ', header=None)
                
                if len(app_index_dict[appliance]) > 1:
                    for idx in app_index_dict[appliance][1:]:
                        temp_data_ = pd.read_csv(house_folder.joinpath(f'channel_{idx}.dat'), sep=' ', header=None)
                        temp_data = pd.merge(temp_data, temp_data_, how='inner', on=0)
                        temp_data.iloc[:, 1] = temp_data.iloc[:,1] + temp_data.iloc[:, 2]
                        temp_data = temp_data.iloc[:, 0:2]
                
                house_data = pd.merge(house_data, temp_data, how='inner', on=0)
                house_data.iloc[:, 0] = pd.to_datetime(house_data.iloc[:, 0], unit='s')
                house_data.columns = ['time', 'aggregate'] + self.appliance_names
                house_data = house_data.set_index('time')
                house_data = house_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)
                
                if entire_data is None:
                    entire_data = house_data
                else:
                    entire_data = pd.concat([entire_data, house_data], ignore_index=True)
        
        if entire_data is None:
            raise ValueError("No valid data found")
        
        entire_data = entire_data.dropna().copy()
        entire_data = entire_data[entire_data['aggregate'] > 0]
        entire_data[entire_data < 5] = 0
        entire_data = entire_data.clip(0, self.cutoff[0], axis=1)
        
        return entire_data.values[:, 0], entire_data.values[:, 1]
    
    def _load_ukdale_data(self):
        """Load and process UK-DALE dataset"""
        directory = Path(self.data_location)
        entire_data = None
        
        for house_id in self.house_indices:
            if house_id not in [1, 2, 3, 4, 5]:
                continue
            
            house_folder = directory.joinpath(f'house_{house_id}')
            house_label = pd.read_csv(house_folder.joinpath('labels.dat'), sep=' ', header=None)
            house_data = pd.read_csv(house_folder.joinpath('channel_1.dat'), sep=' ', header=None)
            
            house_data.columns = ['time', 'aggregate']
            house_data['time'] = pd.to_datetime(house_data['time'], unit='s')
            house_data = house_data.set_index('time').resample(self.sampling).mean().fillna(method='ffill', limit=30)
            
            appliance_list = house_label.iloc[:, 1].values
            app_index_dict = defaultdict(list)
            
            for appliance in self.appliance_names:
                try:
                    idx = appliance_list.tolist().index(appliance)
                    app_index_dict[appliance].append(idx+1)
                except ValueError:
                    app_index_dict[appliance].append(-1)
            
            if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                continue
            
            for appliance in self.appliance_names:
                channel_idx = app_index_dict[appliance][0]
                if channel_idx == -1:
                    house_data.insert(len(house_data.columns), appliance, np.zeros(len(house_data)))
                else:
                    channel_path = house_folder.joinpath(f'channel_{channel_idx}.dat')
                    appl_data = pd.read_csv(channel_path, sep=' ', header=None)
                    appl_data.columns = ['time', appliance]
                    appl_data['time'] = pd.to_datetime(appl_data['time'], unit='s')
                    appl_data = appl_data.set_index('time').resample(self.sampling).mean().fillna(method='ffill', limit=30)
                    house_data = pd.merge(house_data, appl_data, how='inner', left_index=True, right_index=True)
            
            if entire_data is None:
                entire_data = house_data
            else:
                entire_data = pd.concat([entire_data, house_data], ignore_index=True)
        
        if entire_data is None:
            raise ValueError("No valid data found")
        
        entire_data = entire_data.dropna().copy()
        entire_data = entire_data[entire_data['aggregate'] > 0]
        entire_data[entire_data < 5] = 0
        entire_data = entire_data.clip(0, self.cutoff[0], axis=1)
        
        return entire_data.values[:, 0], entire_data.values[:, 1]
    
    def _load_refit_data(self):
        """Load and process REFIT dataset"""
        data_path = Path(self.data_location)
        labels_path = data_path.parent / 'Labels'
        entire_data = None
        
        for house_idx in self.house_indices:
            filename = f'House{house_idx}.csv'
            labelname = f'House{house_idx}.txt'
            house_data_loc = data_path / filename
            
            with open(labels_path / labelname) as f:
                house_labels = f.readlines()
            
            house_labels = ['Time'] + house_labels[0].split(',')
            
            if self.appliance_names[0] in house_labels:
                house_data = pd.read_csv(house_data_loc)
                house_data['Unix'] = pd.to_datetime(house_data['Unix'], unit='s')
                
                house_data = house_data.drop(labels=['Time'], axis=1)
                house_data.columns = house_labels
                house_data = house_data.set_index('Time')
                
                idx_to_drop = house_data[house_data['Issues'] == 1].index
                house_data = house_data.drop(index=idx_to_drop, axis=0)
                house_data = house_data[['Aggregate', self.appliance_names[0]]]
                house_data = house_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)
                
                if entire_data is None:
                    entire_data = house_data
                else:
                    entire_data = pd.concat([entire_data, house_data], ignore_index=True)
        
        if entire_data is None:
            raise ValueError("No valid data found")
        
        entire_data = entire_data.dropna().copy()
        entire_data = entire_data[entire_data['Aggregate'] > 0]
        entire_data[entire_data < 5] = 0
        entire_data = entire_data.clip(0, self.cutoff[0], axis=1)
        
        return entire_data.values[:, 0], entire_data.values[:, 1]
    
    def get_train_datasets(self):
        """Get train and validation datasets"""
        from utils.dataset import NILMDataset
        
        val_end = int(self.val_size * len(self.x))
        
        val = NILMDataset(
            self.x[:val_end],
            self.y[:val_end],
            self.status[:val_end],
            self.window_size,
            self.window_size  # non-overlapping windows for validation
        )
        
        train = NILMDataset(
            self.x[val_end:],
            self.y[val_end:],
            self.status[val_end:],
            self.window_size,
            self.window_stride
        )
        
        return train, val
    
    def get_pretrain_datasets(self, mask_prob=0.25):
        """Get pretrain datasets"""
        from utils.dataset import NILMDataset, Pretrain_Dataset
        
        val_end = int(self.val_size * len(self.x))
        
        val = NILMDataset(
            self.x[:val_end],
            self.y[:val_end],
            self.status[:val_end],
            self.window_size,
            self.window_size
        )
        
        train = Pretrain_Dataset(
            self.x[val_end:],
            self.y[val_end:],
            self.status[val_end:],
            self.window_size,
            self.window_stride,
            mask_prob=mask_prob
        )
        
        return train, val