import os
import torch
import json
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from pathlib import Path
from utils.metrics import (
    acc_precision_recall_f1_score, regression_errors, 
    rmse, energy_accuracy, nde, event_detection_metrics
)
from utils.visualize import NILMVisualizer

class Trainer:
    """
    Trainer class for NILM models.
    Handles training, validation, testing, and visualization.
    """
    
    def __init__(self, args, ds_parser, model, logger=None):
        """
        Initialize trainer with model and data.
        
        Args:
            args: Configuration arguments
            ds_parser: Data processor for dataset
            model: NILM model to train
            logger: Logger object for logging progress
        """
        self.args = args
        self.device = args.device
        self.pretrain = args.pretrain
        self.pretrain_num_epochs = args.pretrain_num_epochs
        self.num_epochs = args.num_epochs
        self.model = model.to(args.device)
        if not os.path.isabs(args.export_root):
            current_file = os.path.abspath(__file__)  # trainer.py的绝对路径
            current_dir = os.path.dirname(current_file)  # trainer.py所在目录 
            project_root = os.path.dirname(current_dir)  # 项目根目录
            export_root = os.path.join(project_root, args.export_root)
        else:
            export_root = args.export_root
            
        self.export_root = Path(export_root).joinpath(args.dataset_code).joinpath(args.appliance_names[0])
        self.best_model_epoch = None
        
        # Logger
        self.logger = logger
        
        # Log initialization info
        self._log_info(f"Initializing trainer, export path: {self.export_root}")
        self._log_info(f"Device: {self.device}, pretrain: {self.pretrain}")
        self._log_info(f"Pretrain epochs: {self.pretrain_num_epochs}, main epochs: {self.num_epochs}")
        
        # Get absolute path to results directory root
        self.results_dir_root = os.path.dirname(os.path.dirname(str(self.export_root)))
        
        # Visualizer
        from utils.visualize import NILMVisualizer  # 确保正确导入
        self.visualizer = NILMVisualizer(
            dataset_code=args.dataset_code, 
            appliance_name=args.appliance_names[0], 
            base_dir=self.results_dir_root
        )

        # Model parameters
        self.cutoff = torch.tensor(args.cutoff[args.appliance_names[0]]).to(self.device)
        self.threshold = torch.tensor(args.threshold[args.appliance_names[0]]).to(self.device)
        self.min_on = torch.tensor(args.min_on[args.appliance_names[0]]).to(self.device)
        self.min_off = torch.tensor(args.min_off[args.appliance_names[0]]).to(self.device)
        self.C0 = torch.tensor(args.c0[args.appliance_names[0]]).to(self.device)
        self.tau = args.tau

        # Log training parameters
        self._log_info(f"Cutoff: {self.cutoff.item()}, threshold: {self.threshold.item()}")
        self._log_info(f"Min on time: {self.min_on.item()}, min off time: {self.min_off.item()}")

        # Import here to avoid circular imports
        from utils.NILM_Dataloader import NILMDataloader

        # Create data loaders - use balanced dataset for training
        if self.pretrain:
            dataloader = NILMDataloader(args, ds_parser, pretrain=True)
            self.pretrain_loader, self.pretrain_val_loader = dataloader.get_dataloaders()
            self._log_info(f"Pretrain dataloaders created, train batches: {len(self.pretrain_loader)}")

        dataloader = NILMDataloader(args, ds_parser, pretrain=False, balance_dataset=True)
        self.train_loader, self.val_loader = dataloader.get_dataloaders()
        self._log_info(f"Main dataloaders created, train batches: {len(self.train_loader)}, val batches: {len(self.val_loader)}")

        # Create optimizer
        self.optimizer = self._create_optimizer()
        self._log_info(f"Using optimizer: {self.args.optimizer}, learning rate: {self.args.lr}")
        
        # Learning rate scheduler
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.decay_step,
                gamma=0.5
            )
            self._log_info(f"Learning rate scheduler enabled, step size: {args.decay_step}, gamma: 0.5")

        # Normalization parameters
        self.normalize = args.normalize
        if self.normalize == 'mean':
            self.x_mean, self.x_std = ds_parser.x_mean, ds_parser.x_std
            self.x_mean = torch.tensor(self.x_mean).to(self.device)
            self.x_std = torch.tensor(self.x_std).to(self.device)
            self._log_info(f"Using mean normalization, mean: {self.x_mean.item()}, std: {self.x_std.item()}")

        # Loss functions
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')

        # Metric tracking dictionaries
        # Per epoch
        self.train_metrics_dict = {
            'mae': [],
            'mre': [],
            'rmse': [],  # New metric
            'eacc': [],  # New metric
            'nde': [],   # New metric
            'acc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'event_tp': [],  # New metric
            'event_fp': [],  # New metric
            'event_fn': []   # New metric
        }
        # Per validate() run
        self.val_metrics_dict = {
            'mae': [],
            'mre': [],
            'rmse': [],
            'eacc': [],
            'nde': [],
            'acc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'event_tp': [],
            'event_fp': [],
            'event_fn': []
        }
        # Test set
        self.test_metrics_dict = {
            'mae': [],
            'mre': [],
            'rmse': [],
            'eacc': [],
            'nde': [],
            'acc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'event_tp': [],
            'event_fp': [],
            'event_fn': []
        }

        # Training history
        self.training_loss = []
        self.y_pred_curve, self.y_curve, self.s_pred_curve, self.status_curve = [], [], [], []

        # Create export directory
        os.makedirs(self.export_root, exist_ok=True)
        self._log_info(f"Export directory created: {self.export_root}")
    
    def _log_info(self, message):
        """Log info message to log file"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _log_warn(self, message):
        """Log warning message to log file"""
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"Warning: {message}")
    
    def _log_error(self, message):
        """Log error message to log file"""
        if self.logger:
            self.logger.error(message)
        else:
            print(f"Error: {message}")

    def train(self):
        """
        Train model, including optional pretraining phase.
        """
        # Initial validation
        _, best_mre, best_acc, _, _, best_f1, _, _, _ = self.validate()
        self._save_state_dict()
        self._log_info(f"Initial validation metrics - MRE: {best_mre:.4f}, Accuracy: {best_acc:.4f}, F1: {best_f1:.4f}")
        
        # Pretraining phase
        if self.pretrain:
            self._log_info("Starting pretraining phase...")
            for epoch in range(self.pretrain_num_epochs):
                self.pretrain_one_epoch(epoch + 1)
            self._log_info("Pretraining completed")

        # Main training phase
        self._log_info("Starting main training phase...")
        self.model.pretrain = False
        for epoch in range(self.num_epochs):
            # Train for one epoch
            self.train_one_epoch(epoch + 1)
            
            # Validate
            mae, mre, acc, precision, recall, f1, rmse_val, eacc_val, nde_val = self.validate()
            
            # Update metrics dictionary
            event_tp, event_fp, event_fn = None, None, None  # Will be calculated during validation
            self.update_metrics_dict(
                mae, mre, rmse_val, eacc_val, nde_val, 
                acc, precision, recall, f1, 
                event_tp, event_fp, event_fn, mode='train'
            )
            
            self._log_info(f"Epoch {epoch+1}/{self.num_epochs} - MRE: {mre:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")

            # Save model if it's the best so far
            combined_score = f1 + acc - mre
            best_combined_score = best_f1 + best_acc - best_mre
            
            if combined_score > best_combined_score:
                self._log_info(f"Found better model - Previous metrics: (MRE={best_mre:.4f}, ACC={best_acc:.4f}, F1={best_f1:.4f})")
                self._log_info(f"New metrics: (MRE={mre:.4f}, ACC={acc:.4f}, F1={f1:.4f})")
                best_f1 = f1
                best_acc = acc
                best_mre = mre
                self.best_model_epoch = epoch
                self._save_state_dict()
        
        self._log_info(f"Training completed, best model at epoch {self.best_model_epoch+1}")
        self._log_info(f"Best model metrics - MRE: {best_mre:.4f}, Accuracy: {best_acc:.4f}, F1: {best_f1:.4f}")
        
        # Generate training visualizations
        self.visualize_training()

    def pretrain_one_epoch(self, epoch):
        """
        Perform one pretraining epoch.
        
        Args:
            epoch: Current epoch number
        """
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.pretrain_loader, desc=f"Pretrain Epoch {epoch}/{self.pretrain_num_epochs}")
        
        for _, batch in enumerate(tqdm_dataloader):
            x, y, status = [batch[i].to(self.device) for i in range(3)]
            self.optimizer.zero_grad()

            # Create mask where status is -1 (masked)
            mask = (status >= 0)

            # Cap values to cutoff
            y_capped = y / self.cutoff

            # Forward pass
            logits, gen_out = self.model(x, mask)
            
            # Reshape for loss function
            logits_masked = torch.masked_select(logits, mask).view((-1))
            labels_masked = torch.masked_select(y_capped, mask).view((-1))
            gen_out = gen_out.view(-1) if gen_out is not None else None

            mask = mask.view(-1).type(torch.DoubleTensor).to(self.device)

            # Calculate loss
            total_loss = self.loss_fn_pretrain(logits_masked, labels_masked, gen_out, mask)

            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()

            # Log loss
            loss_values.append(total_loss.item())
            average_loss = np.mean(np.array(loss_values))
            self.training_loss.append(average_loss)
            tqdm_dataloader.set_description(f'Pretrain Epoch {epoch}/{self.pretrain_num_epochs}, loss {average_loss:.4f}')

        self._log_info(f"Pretrain Epoch {epoch}/{self.pretrain_num_epochs}, average loss: {average_loss:.4f}")
            
        # Update learning rate if scheduled
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()
            self._log_info(f"Learning rate updated to: {self.optimizer.param_groups[0]['lr']}")

    def train_one_epoch(self, epoch):
        """
        Perform one main training epoch.
        
        Args:
            epoch: Current epoch number
        """
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader, desc=f"Training Epoch {epoch}/{self.num_epochs}")
        
        for _, batch in enumerate(tqdm_dataloader):
            x, y, status = [batch[i].to(self.device) for i in range(3)]
            self.optimizer.zero_grad()
            
            # Cap values to cutoff
            y_capped = y / self.cutoff

            # Forward pass
            logits, _ = self.model(x)
            
            # Apply cutoff and compute status
            logits_y = self.cutoff_energy(logits * self.cutoff)
            logits_status = self.compute_status(logits_y)
            
            # Calculate loss
            total_loss = self.loss_fn_train(logits, y_capped, logits_status, status)

            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()
            
            # Log loss
            loss_values.append(total_loss.item())
            average_loss = np.mean(np.array(loss_values))
            self.training_loss.append(average_loss)
            tqdm_dataloader.set_description(f'Training Epoch {epoch}/{self.num_epochs}, loss {average_loss:.4f}')

        self._log_info(f"Training Epoch {epoch}/{self.num_epochs}, average loss: {average_loss:.4f}")
            
        # Update learning rate if scheduled
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()
            self._log_info(f"Learning rate updated to: {self.optimizer.param_groups[0]['lr']}")

    def validate(self):
        """
        Validate model performance.
        
        Returns:
            metrics: Tuple of evaluation metrics
        """
        self.model.eval()
        self.val_metrics_dict = {
            'mae': [],
            'mre': [],
            'rmse': [],
            'eacc': [],
            'nde': [],
            'acc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'event_tp': [],
            'event_fp': [],
            'event_fn': []
        }

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader, desc="Validating")
            for _, batch in enumerate(tqdm_dataloader):
                x, y, status = [batch[i].to(self.device) for i in range(3)]
                
                # Cap values to cutoff
                y_capped = y / self.cutoff

                # Forward pass
                logits, _ = self.model(x)
                
                # Apply cutoff and compute status
                logits_y = self.cutoff_energy(logits * self.cutoff)
                logits_status = self.compute_status(logits_y)
                
                # Apply ON/OFF status to energy prediction
                logits_y = logits_y * logits_status

                # Calculate metrics
                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status, status)
                mae, mre = regression_errors(logits_y, y_capped)
                
                # New metrics
                rmse_val = rmse(logits_y, y_capped)
                eacc_val = energy_accuracy(logits_y, y_capped)
                nde_val = nde(logits_y, y_capped)
                event_tp, event_fp, event_fn = event_detection_metrics(status, logits_status)
                
                # Update metrics dictionary
                self.update_metrics_dict(
                    mae, mre, rmse_val, eacc_val, nde_val, 
                    acc, precision, recall, f1,
                    event_tp, event_fp, event_fn, mode='val'
                )

                # Update progress bar description
                acc_mean = np.mean(np.concatenate(self.val_metrics_dict['acc']).reshape(-1))
                f1_mean = np.mean(np.concatenate(self.val_metrics_dict['f1']).reshape(-1))
                mre_mean = np.mean(np.concatenate(self.val_metrics_dict['mre']).reshape(-1))
                tqdm_dataloader.set_description(f'Validation, rel_err {mre_mean:.4f}, acc {acc_mean:.4f}, f1 {f1_mean:.4f}')

        # Calculate average metrics
        mae_mean = np.mean(np.concatenate(self.val_metrics_dict['mae']).reshape(-1))
        mre_mean = np.mean(np.concatenate(self.val_metrics_dict['mre']).reshape(-1))
        acc_mean = np.mean(np.concatenate(self.val_metrics_dict['acc']).reshape(-1))
        prec_mean = np.mean(np.concatenate(self.val_metrics_dict['precision']).reshape(-1))
        recall_mean = np.mean(np.concatenate(self.val_metrics_dict['recall']).reshape(-1))
        f1_mean = np.mean(np.concatenate(self.val_metrics_dict['f1']).reshape(-1))
        rmse_mean = np.mean(np.concatenate(self.val_metrics_dict['rmse']).reshape(-1))
        eacc_mean = np.mean(np.concatenate(self.val_metrics_dict['eacc']).reshape(-1))
        nde_mean = np.mean(np.concatenate(self.val_metrics_dict['nde']).reshape(-1))
        
        self._log_info(f"Validation results - MRE: {mre_mean:.4f}, ACC: {acc_mean:.4f}, F1: {f1_mean:.4f}, MAE: {mae_mean:.4f}")
        self._log_info(f"Additional metrics - RMSE: {rmse_mean:.4f}, EACC: {eacc_mean:.4f}, NDE: {nde_mean:.4f}")
        
        return mae_mean, mre_mean, acc_mean, prec_mean, recall_mean, f1_mean, rmse_mean, eacc_mean, nde_mean

    def test(self, test_loader):
        """
        Test model performance.
        
        Args:
            test_loader: DataLoader with test data
            
        Returns:
            metrics: Tuple of test metrics
        """
        # Load best model
        self._load_best_model()
        self.model.eval()
        
        # Initialize results arrays
        y_pred_curve, y_curve, s_pred_curve, status_curve = [], [], [], []
        
        self._log_info("Starting testing...")

        with torch.no_grad():
            tqdm_dataloader = tqdm(test_loader, desc="Testing")
            for _, batch in enumerate(tqdm_dataloader):
                x, y, status = [batch[i].to(self.device) for i in range(3)]
        
                # Cap values to cutoff
                y_capped = y / self.cutoff

                # Forward pass
                logits, _ = self.model(x)
                
                # Apply cutoff and compute status
                logits_y = self.cutoff_energy(logits * self.cutoff)
                logits_status = self.compute_status(logits_y)
                
                # Apply ON/OFF status to energy prediction
                logits_y = logits_y * logits_status

                # Calculate metrics
                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status, status)
                mae, mre = regression_errors(logits_y, y_capped)
                
                # New metrics
                rmse_val = rmse(logits_y, y_capped)
                eacc_val = energy_accuracy(logits_y, y_capped)
                nde_val = nde(logits_y, y_capped)
                event_tp, event_fp, event_fn = event_detection_metrics(status, logits_status)
                
                # Update metrics dictionary
                self.update_metrics_dict(
                    mae, mre, rmse_val, eacc_val, nde_val, 
                    acc, precision, recall, f1,
                    event_tp, event_fp, event_fn, mode='test'
                )

                # Update progress bar description
                acc_mean = np.mean(np.concatenate(self.test_metrics_dict['acc']).reshape(-1))
                f1_mean = np.mean(np.concatenate(self.test_metrics_dict['f1']).reshape(-1))
                mre_mean = np.mean(np.concatenate(self.test_metrics_dict['mre']).reshape(-1))
                tqdm_dataloader.set_description(f'Testing, rel_err {mre_mean:.4f}, acc {acc_mean:.4f}, f1 {f1_mean:.4f}')

                # Store results for visualization
                y_pred_curve.append(logits_y.detach().cpu().numpy().squeeze())
                y_curve.append(y.detach().cpu().numpy().squeeze())
                s_pred_curve.append(logits_status.detach().cpu().numpy().squeeze())
                status_curve.append(status.detach().cpu().numpy().squeeze())
            
            # Concatenate results
            self.y_pred_curve = np.concatenate(y_pred_curve).reshape(1, -1)
            self.y_curve = np.concatenate(y_curve).reshape(1, -1)
            self.s_pred_curve = np.concatenate(s_pred_curve).reshape(1, -1)
            self.status_curve = np.concatenate(status_curve).reshape(1, -1)

        # Save test results
        self._save_result({'gt': self.y_curve.tolist(), 'pred': self.y_pred_curve.tolist()}, 'test_result.json')
        
        # Calculate final metrics
        mre, mae = regression_errors(self.y_pred_curve, self.y_curve)
        acc, precision, recall, f1 = acc_precision_recall_f1_score(self.s_pred_curve, self.status_curve)
        rmse_val = rmse(self.y_pred_curve, self.y_curve)
        eacc_val = energy_accuracy(self.y_pred_curve, self.y_curve)
        nde_val = nde(self.y_pred_curve, self.y_curve)
        
        self._log_info(f"Test results - MRE: {mre[0]:.4f}, ACC: {acc[0]:.4f}, F1: {f1[0]:.4f}, MAE: {mae[0]:.4f}")
        self._log_info(f"Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}")
        self._log_info(f"RMSE: {rmse_val[0]:.4f}, EACC: {eacc_val[0]:.4f}, NDE: {nde_val[0]:.4f}")
        
        # Generate test visualizations
        self.visualize_predictions()
        
        return mre, mae, acc, precision, recall, f1, rmse_val, eacc_val, nde_val

    def visualize_training(self):
        """Generate and save training visualization figures organized by dataset and appliance"""
        self._log_info("Generating training visualizations...")
        
        # Create visualizer with dataset and appliance information
        dataset_code = self.args.dataset_code
        appliance_name = self.args.appliance_names[0]
        
        # Use the absolute path to results directory root
        self.visualizer = NILMVisualizer(
            dataset_code=dataset_code,
            appliance_name=appliance_name,
            base_dir=self.results_dir_root
        )
        
        # Plot training loss
        self.visualizer.plot_training_loss(
            self.training_loss, 
            title=f"Training Loss Evolution: {dataset_code.upper()} - {appliance_name}"
        )
        
        # Plot metrics evolution if we have training metrics
        if hasattr(self, 'train_metrics_dict') and len(self.train_metrics_dict['acc']) > 0:
            self.visualizer.plot_metrics_evolution(
                self.train_metrics_dict, 
                len(self.train_metrics_dict['acc'])
            )
        
        self._log_info(f"Training visualizations completed for {dataset_code} - {appliance_name}")

    def visualize_predictions(self, window_size=1000):
        """
        Visualize model predictions against ground truth, organized by dataset and appliance.
        
        Args:
            window_size: Number of time steps to visualize
        """
        self._log_info("Generating prediction visualizations...")
        
        # Create visualizer with dataset and appliance information
        dataset_code = self.args.dataset_code
        appliance_name = self.args.appliance_names[0]
        
        # Use the absolute path to results directory root
        self.visualizer = NILMVisualizer(
            dataset_code=dataset_code,
            appliance_name=appliance_name,
            base_dir=self.results_dir_root
        )
        
        # Check if prediction data exists
        if not hasattr(self, 'y_curve') or len(self.y_curve) == 0:
            self._log_warn("No prediction data available. Run test() first.")
            return
        
        # Energy prediction plot
        self.visualizer.plot_energy_prediction(
            self.y_curve[0], 
            self.y_pred_curve[0], 
            cutoff=self.cutoff.cpu().numpy(),
            window_size=window_size,
            title=f"Energy Consumption: {dataset_code.upper()} - {appliance_name}"
        )
        
        # State prediction plot
        self.visualizer.plot_state_prediction(
            self.status_curve[0],
            self.s_pred_curve[0],
            window_size=window_size,
            title=f"Device State: {dataset_code.upper()} - {appliance_name}"
        )
        
        # Combined plot
        self.visualizer.plot_combined_prediction(
            self.y_curve[0],
            self.y_pred_curve[0],
            self.status_curve[0],
            self.s_pred_curve[0],
            cutoff=self.cutoff.cpu().numpy(),
            window_size=window_size,
            title=f"Energy and State: {dataset_code.upper()} - {appliance_name}"
        )
        
        # Error analysis plot
        self.visualizer.plot_error_analysis(
            self.y_curve[0],
            self.y_pred_curve[0],
            cutoff=self.cutoff.cpu().numpy(),
            window_size=window_size,
            title=f"Prediction Error: {dataset_code.upper()} - {appliance_name}"
        )
        
        self._log_info(f"Prediction visualizations completed for {dataset_code} - {appliance_name}")

    def _save_state_dict(self):
        """Save model state"""
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        self._log_info('Saving best model...')
        model_path = self.export_root.joinpath('best_acc_model.pth')
        torch.save(self.model.state_dict(), model_path)
        self._log_info(f"Model saved to: {model_path}")

    def update_metrics_dict(self, mae, mre, rmse_val, eacc_val, nde_val, 
                           acc, precision, recall, f1, 
                           event_tp=None, event_fp=None, event_fn=None, mode='val'):
        """
        Update metrics dictionary with evaluation results.
        
        Args:
            mae: Mean Absolute Error
            mre: Mean Relative Error
            rmse_val: Root Mean Square Error
            eacc_val: Energy Accuracy
            nde_val: Normalized Disaggregation Error
            acc: Accuracy
            precision: Precision
            recall: Recall
            f1: F1 Score
            event_tp: Event detection true positive rate
            event_fp: Event detection false positive rate
            event_fn: Event detection false negative rate
            mode: Which metrics dict to update ('train', 'val', or 'test')
        """
        if mode == 'train':
            metrics_dict = self.train_metrics_dict
        elif mode == 'val':
            metrics_dict = self.val_metrics_dict
        else:
            metrics_dict = self.test_metrics_dict
        
        metrics_dict['mae'].append(mae)
        metrics_dict['mre'].append(mre)
        metrics_dict['rmse'].append(rmse_val)
        metrics_dict['eacc'].append(eacc_val)
        metrics_dict['nde'].append(nde_val)
        metrics_dict['acc'].append(acc)
        metrics_dict['precision'].append(precision)
        metrics_dict['recall'].append(recall)
        metrics_dict['f1'].append(f1)
        
        if event_tp is not None:
            metrics_dict['event_tp'].append(event_tp)
        if event_fp is not None:
            metrics_dict['event_fp'].append(event_fp)
        if event_fn is not None:
            metrics_dict['event_fn'].append(event_fn)

    def cutoff_energy(self, data):
        """
        Apply cutoff to energy data.
        
        Args:
            data: Energy prediction data
            
        Returns:
            data: Processed data with cutoff applied
        """
        data[data < 5] = 0
        data = torch.min(data, self.cutoff.double())
        return data
        
    def compute_status(self, data):
        """
        Compute ON/OFF status with improved threshold handling.
        
        Args:
            data: Power data
            
        Returns:
            status: ON/OFF status (1 for ON, 0 for OFF)
        """
        # Apply basic threshold
        status = (data >= self.threshold) * 1
        
        # Apply minimum duration filtering
        status_np = status.detach().cpu().numpy()
        for i in range(status_np.shape[0]):  # For each batch sample
            s = status_np[i, 0, :]  # Extract status sequence
            
            # Find state change points
            change_points = np.where(np.diff(np.pad(s, (1, 1), 'constant')))[0]
            
            # Handle odd number of change points
            if len(change_points) % 2 != 0:
                change_points = change_points[:-1]
            
            # Process each interval defined by two change points
            for j in range(0, len(change_points), 2):
                if j+1 < len(change_points):
                    start, end = change_points[j], change_points[j+1]
                    if s[start] == 1:  # ON state
                        duration = end - start
                        if duration < self.min_on.item():
                            # Duration too short, remove this ON interval
                            s[start:end] = 0
                    else:  # OFF state
                        if j > 0:  # Ensure there's a previous state
                            prev_end = change_points[j-1]
                            duration = start - prev_end
                            if duration < self.min_off.item():
                                # OFF duration too short, connect adjacent ON intervals
                                s[prev_end:start] = 1
            
            status_np[i, 0, :] = s
        
        # Convert processed status back to tensor
        return torch.tensor(status_np, device=self.device)

    def _create_optimizer(self):
        """
        Create optimizer based on configuration.
        
        Returns:
            optimizer: PyTorch optimizer
        """
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if self.args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        elif self.args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=self.args.lr)
        elif self.args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=self.args.momentum)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

    def _load_best_model(self):
        """Load best model for testing"""
        try:
            model_path = self.export_root.joinpath('best_acc_model.pth')
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)
            self._log_info(f"Successfully loaded best model: {model_path}")
        except Exception as e:
            self._log_error(f"Failed to load best model: {str(e)}")
            self._log_warn("Continuing with current model for testing...")
    
    def _save_result(self, data, filename):
        """
        Save results to JSON file.
        
        Args:
            data: Data to save
            filename: Output filename
        """
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)
        self._log_info(f"Results saved to {filepath}")

    def loss_fn_gen(self, logits_masked, labels_masked):
        """
        Generator loss function for pretraining.
        
        Args:
            logits_masked: Model logits for masked positions
            labels_masked: Ground truth for masked positions
            
        Returns:
            loss: Combined MSE and KL divergence loss
        """
        mse_arg_1 = logits_masked.contiguous().view(-1).double()
        mse_arg_2 = labels_masked.contiguous().view(-1).double()
        kl_arg_1 = torch.log(F.softmax(logits_masked.squeeze() / self.tau, dim=-1) + 1e-9)
        kl_arg_2 = F.softmax(labels_masked.squeeze() / self.tau, dim=-1)
        mse_loss = self.mse(mse_arg_1, mse_arg_2)
        kl_loss = self.kl(kl_arg_1, kl_arg_2)
        loss = mse_loss + kl_loss
        return loss
        
    def loss_fn_disc(self, gen_out, mask):
        """
        Discriminator loss function for pretraining.
        
        Args:
            gen_out: Generator output
            mask: Mask indicating where values were replaced
            
        Returns:
            loss: BCE loss for the discriminator
        """
        return self.bceloss(gen_out, mask)

    def loss_fn_pretrain(self, logits_masked, labels_masked, gen_out, mask):
        """
        Combined pretraining loss function.
        
        Args:
            logits_masked: Model logits for masked positions
            labels_masked: Ground truth for masked positions
            gen_out: Generator output
            mask: Mask indicating where values were replaced
            
        Returns:
            loss: Combined generator and discriminator loss
        """
        gen_loss = self.loss_fn_gen(logits_masked, labels_masked)
        disc_loss = self.loss_fn_disc(gen_out, mask)
        return gen_loss + disc_loss
        
    def loss_fn_train(self, logits, labels, logits_status, status):
        """
        Training loss function with improved status weight.
        
        Args:
            logits: Model energy prediction logits
            labels: Ground truth energy labels
            logits_status: Model status prediction
            status: Ground truth status
            
        Returns:
            loss: Combined training loss
        """
        kl_arg_1 = torch.log(F.softmax(logits.squeeze() / 0.1, dim=-1) + 1e-9)
        kl_arg_2 = F.softmax(labels.squeeze() / 0.1, dim=-1)
        mse_arg_1 = logits.contiguous().view(-1).double()
        mse_arg_2 = labels.contiguous().view(-1).double()
        margin_arg_1 = (logits_status * 2 - 1).contiguous().view(-1).double()
        margin_arg_2 = (status * 2 - 1).contiguous().view(-1).double()

        kl_loss = self.kl(kl_arg_1, kl_arg_2)
        mse_loss = self.mse(mse_arg_1, mse_arg_2)
        margin_loss = self.margin(margin_arg_1, margin_arg_2)

        # Increase weight for status prediction loss to improve F1 score
        status_weight = 5.0  # Higher value to focus more on status prediction
        total_loss = kl_loss + mse_loss + status_weight * margin_loss
        
        # Special loss for ON states to improve F1 score
        on_mask = ((status == 1) + (status != logits_status.reshape(status.shape))) >= 1
        if on_mask.sum() > 0:
            total_size = torch.tensor(on_mask.shape).prod()
            logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
            labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
            loss_l1_on = self.l1_on(logits_on.contiguous().view(-1), labels_on.contiguous().view(-1))
            # Increase weight for ON states
            total_loss += self.C0 * loss_l1_on / total_size * 2.0  # Doubled weight for ON states
        
        return total_loss