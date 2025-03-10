import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
from pathlib import Path

class NILMVisualizer:
    """
    Visualization class for NILM model results and data.
    Creates publication-quality figures organized by dataset and appliance.
    """
    
    def __init__(self, dataset_code=None, appliance_name=None, base_dir="results", set_style=True):
        """
        Initialize the visualizer.
        
        Args:
            dataset_code: Dataset name/code (e.g., 'redd_lf', 'uk_dale', 'refit')
            appliance_name: Appliance name (e.g., 'refrigerator', 'washer_dryer')
            base_dir: Base directory for saving results
            set_style: Whether to set the default style
        """
        self.dataset_code = dataset_code
        self.appliance_name = appliance_name
        self.base_dir = base_dir
        
        # Set up save directory structure
        if dataset_code and appliance_name:
            self.save_dir = os.path.join(base_dir, dataset_code, appliance_name)
        elif dataset_code:
            self.save_dir = os.path.join(base_dir, dataset_code)
        else:
            self.save_dir = base_dir
            
        # Create directories
        if self.save_dir:
            os.makedirs(os.path.join(self.save_dir, 'figures'), exist_ok=True)
        
        if set_style:
            self.set_style()
    
    def set_style(self):
        """Set the default matplotlib style for consistent visualization"""
        plt.style.use('seaborn-whitegrid')
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.titlesize'] = 14
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['figure.dpi'] = 300
    
    def save_figure(self, fig, filename, subfolder=None):
        """
        Save figure to the specified directory in multiple formats.
        
        Args:
            fig: Matplotlib figure object
            filename: Base filename without extension
            subfolder: Optional subfolder within figures directory
        """
        if self.save_dir:
            fig_dir = os.path.join(self.save_dir, 'figures')
            if subfolder:
                fig_dir = os.path.join(fig_dir, subfolder)
            os.makedirs(fig_dir, exist_ok=True)
            
            # Save in multiple formats for publication
            fig.savefig(os.path.join(fig_dir, f"{filename}.png"), bbox_inches='tight')
            fig.savefig(os.path.join(fig_dir, f"{filename}.pdf"), bbox_inches='tight')
            print(f"Figure saved to {os.path.join(fig_dir, filename)}.png/pdf")
    
    def plot_training_loss(self, loss_values, title=None, xlabel='Iterations', ylabel='Loss'):
        """
        Plot training loss curve.
        
        Args:
            loss_values: List of loss values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_values, 'b-', linewidth=2)
        
        # Set title based on dataset and appliance if not provided
        if title is None and self.dataset_code and self.appliance_name:
            title = f"Training Loss: {self.dataset_code.upper()} - {self.appliance_name}"
        
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        # Save with classification structure
        self.save_figure(fig, f"{self.dataset_code}_{self.appliance_name}_training_loss" 
                        if self.dataset_code and self.appliance_name else "training_loss")
        return fig
    
    def plot_metrics_evolution(self, metrics_dict, num_epochs):
        """
        Plot the evolution of multiple metrics over training epochs.
        
        Args:
            metrics_dict: Dictionary of metric arrays
            num_epochs: Number of training epochs
            
        Returns:
            fig: Matplotlib figure object
        """
        epochs = range(1, num_epochs+1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy and F1 score
        ax1 = axes[0, 0]
        acc_values = [np.mean(epoch_data) for epoch_data in metrics_dict['acc']]
        f1_values = [np.mean(epoch_data) for epoch_data in metrics_dict['f1']]
        ax1.plot(epochs, acc_values, 'b-', linewidth=2, label='Accuracy')
        ax1.plot(epochs, f1_values, 'r-', linewidth=2, label='F1 Score')
        
        # Set title with dataset and appliance info
        classification_title = f"{self.dataset_code.upper()} - {self.appliance_name}: " if self.dataset_code and self.appliance_name else ""
        ax1.set_title(f'{classification_title}Classification Performance')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Precision and recall
        ax2 = axes[0, 1]
        precision_values = [np.mean(epoch_data) for epoch_data in metrics_dict['precision']]
        recall_values = [np.mean(epoch_data) for epoch_data in metrics_dict['recall']]
        ax2.plot(epochs, precision_values, 'g-', linewidth=2, label='Precision')
        ax2.plot(epochs, recall_values, 'y-', linewidth=2, label='Recall')
        ax2.set_title(f'{classification_title}Precision and Recall')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # MAE (Mean Absolute Error)
        ax3 = axes[1, 0]
        mae_values = [np.mean(epoch_data) for epoch_data in metrics_dict['mae']]
        ax3.plot(epochs, mae_values, 'c-', linewidth=2)
        ax3.set_title(f'{classification_title}Mean Absolute Error')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MAE')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # MRE (Mean Relative Error)
        ax4 = axes[1, 1]
        mre_values = [np.mean(epoch_data) for epoch_data in metrics_dict['mre']]
        ax4.plot(epochs, mre_values, 'm-', linewidth=2)
        ax4.set_title(f'{classification_title}Mean Relative Error')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MRE')
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        # Save with classification structure
        self.save_figure(fig, f"{self.dataset_code}_{self.appliance_name}_training_metrics" 
                        if self.dataset_code and self.appliance_name else "training_metrics")
        return fig
    
    def plot_energy_prediction(self, y_true, y_pred, cutoff=None, window_size=1000, 
                              title=None, xlabel='Time Steps', ylabel='Power (W)'):
        """
        Plot energy prediction vs ground truth.
        
        Args:
            y_true: Ground truth energy values
            y_pred: Predicted energy values
            cutoff: Optional cutoff value to scale data
            window_size: Number of time steps to show
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            fig: Matplotlib figure object
        """
        # Get sample data
        total_len = len(y_true)
        sample_len = min(window_size, total_len)
        
        # Choose a random segment or from beginning
        if total_len > sample_len:
            start_idx = np.random.randint(0, total_len - sample_len)
        else:
            start_idx = 0
        
        end_idx = start_idx + sample_len
        
        # Sample data
        y_true_sample = y_true[start_idx:end_idx]
        y_pred_sample = y_pred[start_idx:end_idx]
        
        # Scale if needed
        if cutoff is not None:
            if isinstance(cutoff, torch.Tensor):
                cutoff = cutoff.item()
            y_true_sample = y_true_sample * cutoff
            y_pred_sample = y_pred_sample * cutoff
        
        # Time axis
        time_axis = np.arange(sample_len)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_axis, y_true_sample, 'b-', linewidth=1.5, label='Ground Truth')
        ax.plot(time_axis, y_pred_sample, 'r-', linewidth=1.5, label='Predicted')
        
        # Set title with dataset and appliance info if not provided
        if title is None and self.dataset_code and self.appliance_name:
            title = f"Energy Prediction: {self.dataset_code.upper()} - {self.appliance_name}"
            
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        # Save with classification structure
        self.save_figure(fig, f"{self.dataset_code}_{self.appliance_name}_energy_prediction" 
                        if self.dataset_code and self.appliance_name else "energy_prediction")
        return fig
    
    def plot_state_prediction(self, status_true, status_pred, window_size=1000,
                             title=None, xlabel='Time Steps', ylabel='State (ON/OFF)'):
        """
        Plot device state prediction vs ground truth.
        
        Args:
            status_true: Ground truth status values
            status_pred: Predicted status values
            window_size: Number of time steps to show
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            fig: Matplotlib figure object
        """
        # Get sample data
        total_len = len(status_true)
        sample_len = min(window_size, total_len)
        
        # Choose a random segment or from beginning
        if total_len > sample_len:
            start_idx = np.random.randint(0, total_len - sample_len)
        else:
            start_idx = 0
        
        end_idx = start_idx + sample_len
        
        # Sample data
        status_true_sample = status_true[start_idx:end_idx]
        status_pred_sample = status_pred[start_idx:end_idx]
        
        # Time axis
        time_axis = np.arange(sample_len)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_axis, status_true_sample, 'b-', linewidth=1.5, label='Ground Truth')
        ax.plot(time_axis, status_pred_sample, 'r-', linewidth=1.5, label='Predicted')
        
        # Set title with dataset and appliance info if not provided
        if title is None and self.dataset_code and self.appliance_name:
            title = f"State Prediction: {self.dataset_code.upper()} - {self.appliance_name}"
            
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['OFF', 'ON'])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        # Save with classification structure
        self.save_figure(fig, f"{self.dataset_code}_{self.appliance_name}_state_prediction" 
                        if self.dataset_code and self.appliance_name else "state_prediction")
        return fig
    
    def plot_combined_prediction(self, y_true, y_pred, status_true, status_pred, 
                                cutoff=None, window_size=1000, title=None):
        """
        Plot combined energy and state predictions.
        
        Args:
            y_true: Ground truth energy values
            y_pred: Predicted energy values
            status_true: Ground truth status values
            status_pred: Predicted status values
            cutoff: Optional cutoff value to scale data
            window_size: Number of time steps to show
            title: Plot title
            
        Returns:
            fig: Matplotlib figure object
        """
        # Get sample data
        total_len = len(y_true)
        sample_len = min(window_size, total_len)
        
        # Choose a random segment or from beginning
        if total_len > sample_len:
            start_idx = np.random.randint(0, total_len - sample_len)
        else:
            start_idx = 0
        
        end_idx = start_idx + sample_len
        
        # Sample data
        y_true_sample = y_true[start_idx:end_idx]
        y_pred_sample = y_pred[start_idx:end_idx]
        status_true_sample = status_true[start_idx:end_idx]
        status_pred_sample = status_pred[start_idx:end_idx]
        
        # Scale if needed
        if cutoff is not None:
            if isinstance(cutoff, torch.Tensor):
                cutoff = cutoff.item()
            y_true_sample = y_true_sample * cutoff
            y_pred_sample = y_pred_sample * cutoff
        
        # Time axis
        time_axis = np.arange(sample_len)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Set title with dataset and appliance info if not provided
        if title is None and self.dataset_code and self.appliance_name:
            title = f"Combined Prediction: {self.dataset_code.upper()} - {self.appliance_name}"
        
        # Energy subplot
        ax1.plot(time_axis, y_true_sample, 'b-', linewidth=1.5, label='Ground Truth')
        ax1.plot(time_axis, y_pred_sample, 'r-', linewidth=1.5, label='Predicted')
        if title:
            ax1.set_title(title)
        ax1.set_ylabel('Power (W)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Status subplot
        ax2.plot(time_axis, status_true_sample, 'b-', linewidth=1.5, label='True State')
        ax2.plot(time_axis, status_pred_sample, 'r-', linewidth=1.5, label='Predicted State')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('State')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['OFF', 'ON'])
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        # Save with classification structure
        self.save_figure(fig, f"{self.dataset_code}_{self.appliance_name}_combined_prediction" 
                        if self.dataset_code and self.appliance_name else "combined_prediction")
        return fig
    
    def plot_error_analysis(self, y_true, y_pred, cutoff=None, window_size=1000,
                           title=None, xlabel='Time Steps', ylabel='Absolute Error (W)'):
        """
        Plot error analysis for energy prediction.
        
        Args:
            y_true: Ground truth energy values
            y_pred: Predicted energy values
            cutoff: Optional cutoff value to scale data
            window_size: Number of time steps to show
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            fig: Matplotlib figure object
        """
        # Get sample data
        total_len = len(y_true)
        sample_len = min(window_size, total_len)
        
        # Choose a random segment or from beginning
        if total_len > sample_len:
            start_idx = np.random.randint(0, total_len - sample_len)
        else:
            start_idx = 0
        
        end_idx = start_idx + sample_len
        
        # Sample data
        y_true_sample = y_true[start_idx:end_idx]
        y_pred_sample = y_pred[start_idx:end_idx]
        
        # Calculate error
        if cutoff is not None:
            if isinstance(cutoff, torch.Tensor):
                cutoff = cutoff.item()
            error = np.abs(y_true_sample - y_pred_sample) * cutoff
        else:
            error = np.abs(y_true_sample - y_pred_sample)
        
        # Time axis
        time_axis = np.arange(sample_len)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_axis, error, 'r-', linewidth=1.5)
        ax.axhline(y=np.mean(error), color='k', linestyle='--', 
                   label=f'Mean: {np.mean(error):.2f}')
        
        # Set title with dataset and appliance info if not provided
        if title is None and self.dataset_code and self.appliance_name:
            title = f"Error Analysis: {self.dataset_code.upper()} - {self.appliance_name}"
            
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        # Save with classification structure
        self.save_figure(fig, f"{self.dataset_code}_{self.appliance_name}_error_analysis" 
                        if self.dataset_code and self.appliance_name else "error_analysis")
        return fig
    
    def plot_dataset_comparison(self, dataset_metrics, metric_name='f1', appliance_name=None):
        """
        Plot comparison of a specific metric across different datasets.
        
        Args:
            dataset_metrics: Dictionary with dataset names as keys and metric values as values
            metric_name: Name of the metric to compare
            appliance_name: Name of the appliance for the title
            
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort datasets by performance
        sorted_datasets = sorted(dataset_metrics.items(), key=lambda x: x[1], reverse=True)
        datasets = [item[0] for item in sorted_datasets]
        values = [item[1] for item in sorted_datasets]
        
        # Bar plot
        bars = ax.bar(datasets, values, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Title and labels
        metric_display = metric_name.upper() if metric_name in ['mae', 'mre', 'rmse'] else metric_name.capitalize()
        title = f"{metric_display} Comparison Across Datasets"
        if appliance_name:
            title += f" for {appliance_name}"
        
        ax.set_title(title)
        ax.set_xlabel('Dataset')
        ax.set_ylabel(metric_display)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        plt.tight_layout()
        # Save in comparison folder
        subfolder = "comparisons"
        filename = f"{appliance_name}_{metric_name}_dataset_comparison" if appliance_name else f"{metric_name}_dataset_comparison"
        self.save_figure(fig, filename, subfolder=subfolder)
        return fig
    
    def plot_appliance_comparison(self, appliance_metrics, metric_name='f1', dataset_name=None):
        """
        Plot comparison of a specific metric across different appliances.
        
        Args:
            appliance_metrics: Dictionary with appliance names as keys and metric values as values
            metric_name: Name of the metric to compare
            dataset_name: Name of the dataset for the title
            
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sort appliances by performance
        sorted_appliances = sorted(appliance_metrics.items(), key=lambda x: x[1], reverse=True)
        appliances = [item[0] for item in sorted_appliances]
        values = [item[1] for item in sorted_appliances]
        
        # Bar plot
        bars = ax.bar(appliances, values, color='lightgreen')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Rotate x-axis labels for better readability with many appliances
        plt.xticks(rotation=45, ha='right')
        
        # Title and labels
        metric_display = metric_name.upper() if metric_name in ['mae', 'mre', 'rmse'] else metric_name.capitalize()
        title = f"{metric_display} Comparison Across Appliances"
        if dataset_name:
            title += f" for {dataset_name.upper()}"
        
        ax.set_title(title)
        ax.set_xlabel('Appliance')
        ax.set_ylabel(metric_display)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        plt.tight_layout()
        # Save in comparison folder
        subfolder = "comparisons"
        filename = f"{dataset_name}_{metric_name}_appliance_comparison" if dataset_name else f"{metric_name}_appliance_comparison"
        self.save_figure(fig, filename, subfolder=subfolder)
        return fig