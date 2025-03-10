import torch
torch.set_default_tensor_type(torch.DoubleTensor)

import os
import pickle
from pathlib import Path
from time import time

from utils.config import get_args, setup_seed
from utils.data_processor import DataProcessor
from models.Electricity_model import ELECTRICITY
from utils.NILM_Dataloader import NILMDataloader
from trainer import Trainer
from utils.logger import setup_logger

def main():
    # Get command line arguments
    args = get_args()
    
    # Setup logger
    logger = setup_logger(
        name="train",
        dataset_code=args.dataset_code,
        appliance_name=args.appliance_names[0]
    )
    
    logger.info(f"Starting training for {args.dataset_code} dataset, appliance: {args.appliance_names[0]}")
    logger.info(f"Parameters: batch_size={args.batch_size}, epochs={args.num_epochs}, pretrain={args.pretrain}")
    
    # Setup data processor
    logger.info(f"Processing {args.dataset_code} dataset...")
    data_processor = DataProcessor(args)
    
    # Create model
    logger.info("Creating model...")
    model = ELECTRICITY(args)
    logger.info(f"Model structure created, hidden size: {args.hidden}, layers: {args.n_layers}")
    
    # Create trainer
    logger.info("Setting up trainer...")
    trainer = Trainer(args, data_processor, model, logger)
    
    # Training loop
    logger.info("Starting training...")
    start_time = time()
    if args.num_epochs > 0:
        try:
            model_path = os.path.join(trainer.export_root, 'best_acc_model.pth')
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            logger.info('Successfully loaded previous model, continuing training...')
        except FileNotFoundError:
            logger.info('Previous model not found, training new model...')
        trainer.train()
    
    end_time = time()
    training_time = end_time - start_time
    logger.info(f"Total training time: {training_time/60:.2f} minutes")
    
    # Testing loop
    logger.info("Starting testing...")
    args.validation_size = 1.0
    x_mean = trainer.x_mean.detach().cpu().numpy()
    x_std = trainer.x_std.detach().cpu().numpy()
    stats = (x_mean, x_std)
    
    # Set test house indices based on dataset
    if args.dataset_code == 'redd_lf':
        args.house_indicies = [1]
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [2]
    elif args.dataset_code == 'refit':
        args.house_indicies = [5]
    
    logger.info(f"Test houses: {args.house_indicies}")
    
    # Create test data processor and dataloader
    test_processor = DataProcessor(args)
    dataloader = NILMDataloader(args, test_processor)
    _, test_loader = dataloader.get_dataloaders()
    
    # Check if test dataset is valid
    # Check if test dataset is valid
    if len(test_loader) == 0:
        logger.error(f"Test dataset for {args.appliance_names[0]} is empty or too small for the window size.")
        logger.info("Testing skipped. Try with a different house or appliance.")
        # Save partial results without testing metrics
        results = dict()
        results['args'] = args
        results['training_time'] = training_time/60
        results['best_epoch'] = trainer.best_model_epoch
        results['training_loss'] = trainer.training_loss
        # Save other available results...
        
        fname = trainer.export_root.joinpath('results.pkl')
        with open(fname, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Partial results saved to {fname}")
        logger.info("Training completed but testing skipped due to insufficient data")
    else:
        # Perform testing only if test_loader has data
        mre, mae, acc, prec, recall, f1, rmse_val, eacc_val, nde_val = trainer.test(test_loader)
    logger.info(f'Mean Accuracy: {acc}')
    logger.info(f'Mean F1-Score: {f1}')
    logger.info(f'MAE: {mae}')
    logger.info(f'MRE: {mre}')
    
    # Save results
    results = dict()
    results['args'] = args
    results['training_time'] = training_time/60
    results['best_epoch'] = trainer.best_model_epoch
    results['training_loss'] = trainer.training_loss
    results['val_rel_err'] = trainer.test_metrics_dict['mre']
    results['val_abs_err'] = trainer.test_metrics_dict['mae']
    results['val_acc'] = trainer.test_metrics_dict['acc']
    results['val_precision'] = trainer.test_metrics_dict['precision']
    results['val_recall'] = trainer.test_metrics_dict['recall']
    results['val_f1'] = trainer.test_metrics_dict['f1']
    # Add new metrics
    results['val_rmse'] = trainer.test_metrics_dict['rmse']
    results['val_eacc'] = trainer.test_metrics_dict['eacc']
    results['val_nde'] = trainer.test_metrics_dict['nde']
    results['label_curve'] = trainer.y_curve
    results['e_pred_curve'] = trainer.y_pred_curve
    results['status_curve'] = trainer.status_curve
    results['s_pred_curve'] = trainer.s_pred_curve
    
    fname = trainer.export_root.joinpath('results.pkl')
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {fname}")
    logger.info("Training and testing completed")

if __name__ == "__main__":
    main()