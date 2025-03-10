"""
NILM Results Comparison Tool

This script compares results across different datasets and appliances.
It automatically reads the results files and generates comparison plots.
"""

import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path

# Import visualizer directly from file to avoid import issues
exec(open("utils/visualize.py").read())

def load_results(results_dir="results"):
    """
    Load all results from the results directory structure.
    
    Args:
        results_dir: Base directory containing results
        
    Returns:
        metrics_collection: Dictionary with format {dataset: {appliance: {metric: value}}}
    """
    metrics_collection = {}
    results_path = Path(results_dir)
    
    # Walk through all dataset directories
    for dataset_dir in [d for d in results_path.iterdir() if d.is_dir() and d.name != "comparisons"]:
        dataset_name = dataset_dir.name
        metrics_collection[dataset_name] = {}
        
        # Walk through all appliance directories in this dataset
        for appliance_dir in [d for d in dataset_dir.iterdir() if d.is_dir()]:
            appliance_name = appliance_dir.name
            
            # Try to load results.pkl
            try:
                with open(appliance_dir / "results.pkl", "rb") as f:
                    results = pickle.load(f)
                
                # Extract metrics from results dictionary
                metrics = {}
                
                # Extract key metrics
                try:
                    metrics['acc'] = float(np.mean(results['val_acc']))
                    metrics['f1'] = float(np.mean(results['val_f1']))
                    metrics['precision'] = float(np.mean(results['val_precision']))
                    metrics['recall'] = float(np.mean(results['val_recall']))
                    metrics['mae'] = float(np.mean(results['val_abs_err']))
                    metrics['mre'] = float(np.mean(results['val_rel_err']))
                    
                    # Check for newer metrics if available
                    if 'val_rmse' in results:
                        metrics['rmse'] = float(np.mean(results['val_rmse']))
                    if 'val_eacc' in results:
                        metrics['eacc'] = float(np.mean(results['val_eacc']))
                    if 'val_nde' in results:
                        metrics['nde'] = float(np.mean(results['val_nde']))
                    
                    print(f"Loaded metrics for {dataset_name}/{appliance_name}")
                    metrics_collection[dataset_name][appliance_name] = metrics
                    
                except KeyError as e:
                    print(f"Warning: Missing metric in {dataset_name}/{appliance_name} results: {e}")
                
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                print(f"Could not load results for {dataset_name}/{appliance_name}: {e}")
                
                # Try to load from test_result.json
                try:
                    with open(appliance_dir / "test_result.json", "r") as f:
                        test_results = json.load(f)
                    
                    # Extract basic metrics if available
                    if 'metrics' in test_results:
                        metrics_collection[dataset_name][appliance_name] = test_results['metrics']
                        print(f"Loaded metrics from test_result.json for {dataset_name}/{appliance_name}")
                except (FileNotFoundError, json.JSONDecodeError):
                    print(f"No valid results found for {dataset_name}/{appliance_name}")
    
    return metrics_collection

def compare_results(metrics_collection, comparison_type='dataset', metric_name='f1', 
                   visualizer=None, save_dir="results/comparisons"):
    """
    Compare performance across different datasets or appliances.
    
    Args:
        metrics_collection: Dictionary with format {dataset_name: {appliance_name: {metric_name: value}}}
        comparison_type: 'dataset' or 'appliance'
        metric_name: Metric to compare ('f1', 'acc', 'mae', etc.)
        visualizer: Optional visualizer instance
        save_dir: Directory to save comparison results
        
    Returns:
        None, saves comparison plots
    """
    if visualizer is None:
        visualizer = NILMVisualizer(base_dir=save_dir)
    
    if comparison_type == 'dataset':
        # Compare performance of same appliance across different datasets
        # Find all unique appliances
        all_appliances = set()
        for dataset in metrics_collection:
            all_appliances.update(metrics_collection[dataset].keys())
        
        for appliance in all_appliances:
            dataset_metrics = {}
            for dataset in metrics_collection:
                if appliance in metrics_collection[dataset]:
                    if metric_name in metrics_collection[dataset][appliance]:
                        dataset_metrics[dataset] = metrics_collection[dataset][appliance][metric_name]
            
            if dataset_metrics:  # Only create plot if we have data
                print(f"Creating dataset comparison for '{appliance}' using {metric_name} metric")
                visualizer.plot_dataset_comparison(dataset_metrics, metric_name, appliance)
    
    elif comparison_type == 'appliance':
        # Compare different appliances within the same dataset
        for dataset in metrics_collection:
            appliance_metrics = {}
            for app in metrics_collection[dataset]:
                if metric_name in metrics_collection[dataset][app]:
                    appliance_metrics[app] = metrics_collection[dataset][app][metric_name]
            
            if appliance_metrics:  # Only create plot if we have data
                print(f"Creating appliance comparison for '{dataset}' using {metric_name} metric")
                visualizer.plot_appliance_comparison(appliance_metrics, metric_name, dataset)
    
    print(f"Comparison visualizations saved to {os.path.join(save_dir, 'figures/comparisons')}")
    return visualizer

def create_comparison_table(metrics_collection, metric_name='f1', save_dir="results/comparisons"):
    """
    Create comprehensive comparison tables in CSV format.
    
    Args:
        metrics_collection: Dictionary with format {dataset_name: {appliance_name: {metric_name: value}}}
        metric_name: Primary metric for sorting
        save_dir: Directory to save tables
        
    Returns:
        None, saves CSV files
    """
    import pandas as pd
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Create dataset-appliance matrix for the primary metric
    datasets = list(metrics_collection.keys())
    
    # Find all unique appliances
    all_appliances = set()
    for dataset in metrics_collection:
        all_appliances.update(metrics_collection[dataset].keys())
    all_appliances = sorted(list(all_appliances))
    
    # Create dataframe
    df = pd.DataFrame(index=datasets, columns=all_appliances)
    
    # Fill with metric values
    for dataset in datasets:
        for appliance in all_appliances:
            if appliance in metrics_collection[dataset] and metric_name in metrics_collection[dataset][appliance]:
                df.loc[dataset, appliance] = metrics_collection[dataset][appliance][metric_name]
            else:
                df.loc[dataset, appliance] = None
    
    # Save as CSV
    df.to_csv(os.path.join(save_dir, f"{metric_name}_comparison_matrix.csv"))
    print(f"Created comparison matrix for {metric_name} metric")
    
    # 2. Create comprehensive dataset-metric table for each appliance
    all_metrics = ['acc', 'f1', 'precision', 'recall', 'mae', 'mre', 'rmse', 'eacc', 'nde']
    
    for appliance in all_appliances:
        # Create multi-index dataframe: dataset × metric
        metrics_to_use = []
        for metric in all_metrics:
            for dataset in datasets:
                if (appliance in metrics_collection[dataset] and 
                    metric in metrics_collection[dataset][appliance]):
                    if metric not in metrics_to_use:
                        metrics_to_use.append(metric)
                    break
        
        if not metrics_to_use:
            continue
            
        df = pd.DataFrame(index=datasets, columns=metrics_to_use)
        
        # Fill with values
        for dataset in datasets:
            if appliance in metrics_collection[dataset]:
                for metric in metrics_to_use:
                    if metric in metrics_collection[dataset][appliance]:
                        df.loc[dataset, metric] = metrics_collection[dataset][appliance][metric]
                    else:
                        df.loc[dataset, metric] = None
        
        # Save as CSV
        df.to_csv(os.path.join(save_dir, f"{appliance}_metrics_comparison.csv"))
        print(f"Created comprehensive metrics table for {appliance}")
    
    # 3. Create comprehensive appliance-metric table for each dataset
    for dataset in datasets:
        appliances_in_dataset = list(metrics_collection[dataset].keys())
        if not appliances_in_dataset:
            continue
            
        metrics_to_use = []
        for metric in all_metrics:
            for appliance in appliances_in_dataset:
                if metric in metrics_collection[dataset][appliance]:
                    if metric not in metrics_to_use:
                        metrics_to_use.append(metric)
                    break
        
        if not metrics_to_use:
            continue
            
        df = pd.DataFrame(index=appliances_in_dataset, columns=metrics_to_use)
        
        # Fill with values
        for appliance in appliances_in_dataset:
            for metric in metrics_to_use:
                if metric in metrics_collection[dataset][appliance]:
                    df.loc[appliance, metric] = metrics_collection[dataset][appliance][metric]
                else:
                    df.loc[appliance, metric] = None
        
        # Save as CSV
        df.to_csv(os.path.join(save_dir, f"{dataset}_appliance_comparison.csv"))
        print(f"Created comprehensive appliance comparison table for {dataset}")

def main():
    """Main function for command-line operation"""
    parser = argparse.ArgumentParser(description='Compare NILM results across datasets and appliances')
    parser.add_argument('--results_dir', type=str, default='results', 
                        help='Directory containing results')
    parser.add_argument('--metric', type=str, default='f1', 
                        choices=['f1', 'acc', 'precision', 'recall', 'mae', 'mre', 'rmse', 'eacc', 'nde'],
                        help='Metric to compare')
    parser.add_argument('--comparison_type', type=str, default='both', 
                        choices=['dataset', 'appliance', 'both'],
                        help='Type of comparison to perform')
    parser.add_argument('--save_dir', type=str, default='results/comparisons', 
                        help='Directory to save comparisons')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    metrics_collection = load_results(args.results_dir)
    
    if not metrics_collection:
        print("No results found. Make sure the directory structure is correct.")
        return
    
    # Create visualizer
    visualizer = NILMVisualizer(base_dir=args.save_dir)
    
    # Perform comparisons
    if args.comparison_type in ['dataset', 'both']:
        compare_results(metrics_collection, 'dataset', args.metric, visualizer, args.save_dir)
    
    if args.comparison_type in ['appliance', 'both']:
        compare_results(metrics_collection, 'appliance', args.metric, visualizer, args.save_dir)
    
    # Create comparison tables
    create_comparison_table(metrics_collection, args.metric, args.save_dir)
    
    print("All comparisons completed successfully!")

if __name__ == "__main__":
    main()