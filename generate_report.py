import json
import matplotlib.pyplot as plt
import os
import glob
import datetime
import numpy as np
import argparse

RESULTS_DIR = 'results'
REPORTS_DIR = 'reports'

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

def generate_report(result_file=None):
    if result_file is None:
        list_of_files = glob.glob(os.path.join(RESULTS_DIR, 'simulation_results_*.json'))
        if not list_of_files:
            print("No results file found. Run fl_simulation.py first.")
            return
        
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = result_file
        if not os.path.exists(latest_file):
            print(f"Error: Specified file not found: {latest_file}")
            return
            
    # Extract timestamp and username from filename if possible
    # FORMAT: simulation_results_{username}_{timestamp}.json
    base_name = os.path.basename(latest_file)
    content_name = base_name.replace('simulation_results_', '').replace('.json', '')
    
    print(f"Generating report for: {latest_file}")

    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    results = data.get('experiments', data) # Fallback if old format
    
    task_type = metadata.get('task_type', 'classification')
    username = metadata.get('username', 'anonymous')
    
    keys = list(results.keys())
    model_experiments = [k for k in keys if '_FedAvg' in k]
    strategy_experiments = [k for k in keys if 'DeepMLP_' in k]
    
    primary_metric = 'accuracy' if task_type == 'classification' else 'rmse'
    primary_metric_label = 'Accuracy (%)' if task_type == 'classification' else 'RMSE'
    
    secondary_metric = 'auc' if task_type == 'classification' else 'mae'
    secondary_metric_label = 'AUC-ROC' if task_type == 'classification' else 'MAE'
    
    rounds = range(1, len(results[keys[0]]['loss']) + 1)
    
    model_plot_file = f'{REPORTS_DIR}/model_comparison_{content_name}.png'
    strategy_plot_file = f'{REPORTS_DIR}/strategy_comparison_{content_name}.png'
    report_file = f'{REPORTS_DIR}/simulation_report_{content_name}.md'
    
    # 1. Plot Model Comparison
    plt.figure(figsize=(10, 6))
    for key in model_experiments:
        label = key.replace('_FedAvg', '')
        if primary_metric in results[key]:
            plt.plot(rounds, results[key][primary_metric], label=label)
    plt.xlabel('Communication Round')
    plt.ylabel(primary_metric_label)
    plt.title(f'Model Architecture Comparison (Strategy: FedAvg)')
    plt.legend()
    plt.grid(True)
    plt.savefig(model_plot_file)
    plt.close()
    
    # 2. Plot Strategy Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for key in strategy_experiments:
        label = key.replace('DeepMLP_', '')
        if primary_metric in results[key]:
            axes[0].plot(rounds, results[key][primary_metric], label=label)
    axes[0].set_xlabel('Communication Round')
    axes[0].set_ylabel(primary_metric_label)
    axes[0].set_title(f'Strategy Convergence ({primary_metric_label})')
    axes[0].legend()
    axes[0].grid(True)
    
    for key in strategy_experiments:
        label = key.replace('DeepMLP_', '')
        if 'loss' in results[key]:
            axes[1].plot(rounds, results[key]['loss'], label=label)
    axes[1].set_xlabel('Communication Round')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'Strategy Loss Convergence')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(strategy_plot_file)
    plt.close()
    
    # 3. Generating text summary
    with open(report_file, 'w') as f:
        f.write(f"# Federated Learning Simulation Report for **{username.capitalize()}**\n\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Task Type**: {task_type.capitalize()}\n")
        f.write(f"**Target**: `{metadata.get('target_col', 'Unknown')}`\n")
        f.write(f"**Clients**: {metadata.get('num_clients', 'N/A')} (Distribution: {metadata.get('distribution', 'N/A')})\n\n")
        
        f.write("## 1. Strategy Comparison Summary (The 'Ups and Downs')\n")
        f.write("Based on the captured metrics, here is an analysis of the federated learning algorithms:\n\n")
        f.write("- **FedAvg**: The standard baseline. Usually provides steady convergence but can be volatile if client data is highly skewed (non-IID).\n")
        f.write("- **FedProx**: Introduces a proximal term to constrain local updates. **Ups:** Offers superior stability and prevents the global model from diverging in highly heterogeneous (non-IID) health datasets. **Downs:** Can converge slightly slower than FedAvg initially.\n")
        f.write("- **FedAdam**: Uses adaptive server-side momentum. **Ups:** Can provide much faster initial convergence and reach a better optimum. **Downs:** Highly sensitive to the learning rate and hyperparameter tuning; can overshoot if not configured perfectly.\n\n")
        
        f.write("## 2. Final Results Table\n\n")
        
        if task_type == 'classification':
            f.write("| Algorithm | Final Acc | Final F1 | Final AUC | Sensitivity | Specificity | Final Loss |\n")
            f.write("|-----------|-----------|----------|-----------|-------------|-------------|------------|\n")
            for key in strategy_experiments:
                name = key.replace('DeepMLP_', '')
                acc = results[key].get('accuracy', [0])[-1]
                f1 = results[key].get('f1', [0])[-1]
                auc = results[key].get('auc', [0])[-1]
                sens = results[key].get('recall', [0])[-1]
                spec = results[key].get('specificity', [0])[-1]
                loss = results[key].get('loss', [0])[-1]
                f.write(f"| **{name}** | {acc:.2f}% | {f1:.4f} | {auc:.4f} | {sens:.4f} | {spec:.4f} | {loss:.4f} |\n")
        else:
            f.write("| Algorithm | Final RMSE | Final MAE | Final Loss |\n")
            f.write("|-----------|------------|-----------|------------|\n")
            for key in strategy_experiments:
                name = key.replace('DeepMLP_', '')
                rmse = results[key].get('rmse', [0])[-1]
                mae = results[key].get('mae', [0])[-1]
                loss = results[key].get('loss', [0])[-1]
                f.write(f"| **{name}** | {rmse:.4f} | {mae:.4f} | {loss:.4f} |\n")
        
        f.write("\n## 3. Visualizations\n\n")
        f.write(f"![Strategy Comparison]({os.path.abspath(strategy_plot_file)})\n\n")
        f.write(f"![Model Comparison]({os.path.abspath(model_plot_file)})\n\n")
        
    print(f"Report securely generated at: {report_file}")
    return {
        'report_file': report_file,
        'model_plot': model_plot_file,
        'strategy_plot': strategy_plot_file,
        'content_name': content_name
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FL Report")
    parser.add_argument('--result_file', type=str, default=None, help="Specific JSON file to parse")
    args = parser.parse_args()
    
    generate_report(args.result_file)
