import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
import os
import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, confusion_matrix
from src.fl_model import SimpleMLP, LogisticRegression, DeepMLP
from src.fl_data_preprocessing import load_and_preprocess_data
from src.fl_strategies import aggregate_fedavg, proximal_term, FedAdamServer

RESULTS_DIR = 'results'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def evaluate(model, test_loader, task_type='classification', num_classes=1):
    model.eval()
    loss_sum = 0
    criterion = nn.CrossEntropyLoss() if task_type == 'classification' else nn.MSELoss()
    
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            if task_type == 'regression':
                labels = labels.view(-1, 1).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * inputs.size(0)
            
            all_targets.extend(labels.cpu().numpy())
            
            if task_type == 'classification':
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs)
            else:
                all_preds.extend(outputs.cpu().numpy())

    avg_loss = loss_sum / len(all_targets)
    metrics = {'loss': avg_loss}

    if task_type == 'classification':
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        metrics['accuracy'] = accuracy_score(all_targets, all_preds) * 100
        
        if num_classes == 2:
            metrics['f1'] = f1_score(all_targets, all_preds)
            metrics['precision'] = precision_score(all_targets, all_preds, zero_division=0)
            metrics['recall'] = recall_score(all_targets, all_preds, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            try:
                metrics['auc'] = roc_auc_score(all_targets, all_probs[:, 1])
            except ValueError:
                metrics['auc'] = 0.5
        else:
            metrics['f1'] = f1_score(all_targets, all_preds, average='macro')
            metrics['precision'] = precision_score(all_targets, all_preds, average='macro', zero_division=0)
            metrics['recall'] = recall_score(all_targets, all_preds, average='macro', zero_division=0)
            metrics['specificity'] = 0.0 # Specificity not computed for macro multi-class
            
            try:
                metrics['auc'] = roc_auc_score(all_targets, all_probs, multi_class='ovr')
            except ValueError:
                metrics['auc'] = 0.5
    else:
        metrics['rmse'] = np.sqrt(mean_squared_error(all_targets, all_preds))
        metrics['mae'] = mean_absolute_error(all_targets, all_preds)

    return metrics

def train_client(model, train_loader, epochs, lr, strategy='FedAvg', global_model=None, task_type='classification', prox_mu=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if task_type == 'classification' else nn.MSELoss()
    
    for _ in range(epochs):
        for inputs, labels in train_loader:
            if task_type == 'regression':
                labels = labels.view(-1, 1).float()
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if strategy == 'FedProx' and global_model is not None:
                loss += proximal_term(model, global_model, prox_mu)
                
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def get_model(model_name, input_dim, output_dim):
    if model_name == 'LogisticRegression':
        return LogisticRegression(input_dim, output_dim)
    elif model_name == 'SimpleMLP':
        return SimpleMLP(input_dim, output_dim)
    elif model_name == 'DeepMLP':
        return DeepMLP(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_simulation(strategy_name, model_name, client_loaders, test_loader, input_dim, output_dim, args, progress_callback=None, stop_event=None):
    print(f"--- Starting Simulation: Strategy={strategy_name}, Model={model_name} ---")
    
    global_model = get_model(model_name, input_dim, output_dim)
    
    fed_adam_server = None
    if strategy_name == 'FedAdam':
        fed_adam_server = FedAdamServer(global_model, lr=0.01) 
    
    metrics_history = {
        'times': []
    }
    
    start_time_sim = time.time()
    
    for round_idx in range(args.num_rounds):
        if stop_event and stop_event.is_set():
            print("Simulation cancelled by user.")
            break

        round_start = time.time()
        
        global_weights = global_model.state_dict()
        client_updates = []
        
        for client_idx, train_loader in enumerate(client_loaders):
            local_model = get_model(model_name, input_dim, output_dim)
            local_model.load_state_dict(copy.deepcopy(global_weights))
            
            global_model_ref = copy.deepcopy(global_model) if strategy_name == 'FedProx' else None
            
            w_local = train_client(
                local_model, train_loader, args.local_epochs, args.lr, 
                strategy_name, global_model_ref, args.task_type, args.proximal_mu
            )
            client_updates.append(w_local)
        
        if strategy_name == 'FedAdam':
            new_state_dict = fed_adam_server.step(global_model, client_updates)
            global_model.load_state_dict(new_state_dict)
        else:
            new_state_dict = aggregate_fedavg(client_updates)
            global_model.load_state_dict(new_state_dict)
        
        round_metrics = evaluate(global_model, test_loader, args.task_type, output_dim)
        
        for k, v in round_metrics.items():
            if k not in metrics_history:
                metrics_history[k] = []
            metrics_history[k].append(float(v))
            
        metrics_history['times'].append(time.time() - round_start)
        
        display_metric = f"Acc: {round_metrics.get('accuracy', 0):.2f}%" if args.task_type == 'classification' else f"RMSE: {round_metrics.get('rmse', 0):.4f}"
        print(f"[{model_name}][{strategy_name}] Round {round_idx+1}/{args.num_rounds} | {display_metric} | Loss: {round_metrics['loss']:.4f}")

        if progress_callback:
            progress_callback({
                'strategy': strategy_name,
                'model': model_name,
                'round': round_idx + 1,
                'total_rounds': args.num_rounds,
                'metrics': {k: float(v) for k, v in round_metrics.items()}
            })

    total_time = time.time() - start_time_sim
    metrics_history['total_sim_time'] = total_time
    print(f"Finished {strategy_name} with {model_name} in {total_time:.2f}s")
    
    return metrics_history

def execute_experiment(args, progress_callback=None, stop_event=None):
    client_data, (X_test, y_test), num_classes, _ = load_and_preprocess_data(
        args.data_path, args.target_col, args.task_type, args.num_clients, args.distribution
    )
    
    if client_data is None:
        print("Dataset load failed.")
        return None, "Dataset load failed."

    client_loaders = []
    y_dtype = torch.long if args.task_type == 'classification' else torch.float32
    
    for X, y in client_data:
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=y_dtype))
        client_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True))
    
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=y_dtype))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    
    input_dim = client_data[0][0].shape[1]
    output_dim = num_classes if args.task_type == 'classification' else 1
    
    strategies = getattr(args, 'strategies', ['FedAvg', 'FedProx', 'FedAdam'])
    models = getattr(args, 'models', ['LogisticRegression', 'SimpleMLP', 'DeepMLP'])
    
    results = {
        'metadata': {
            'username': args.username,
            'data_path': args.data_path,
            'target_col': args.target_col,
            'task_type': args.task_type,
            'distribution': args.distribution,
            'num_clients': args.num_clients,
            'num_rounds': args.num_rounds
        },
        'experiments': {}
    }
    
    print("\n=== Phase 1: Model Comparison (using FedAvg) ===")
    for model_name in models:
        if stop_event and stop_event.is_set():
            break
        key = f"{model_name}_FedAvg"
        results['experiments'][key] = run_simulation('FedAvg', model_name, client_loaders, test_loader, input_dim, output_dim, args, progress_callback, stop_event)

    print("\n=== Phase 2: Strategy Comparison (using DeepMLP) ===")
    target_model = 'DeepMLP' if 'DeepMLP' in models else models[-1]
    for strategy in strategies:
        if stop_event and stop_event.is_set():
            break
        if strategy == 'FedAvg' and f"{target_model}_FedAvg" in results['experiments']:
            continue
        key = f"{target_model}_{strategy}"
        results['experiments'][key] = run_simulation(strategy, target_model, client_loaders, test_loader, input_dim, output_dim, args, progress_callback, stop_event)
        
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Output uniquely tagged by the runner's username
    filename = f'simulation_results_{args.username}_{timestamp}.json'
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {filepath}")
    return filepath, None

def main():
    parser = argparse.ArgumentParser(description="HealthCare Federated Learning Simulation")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument('--target_col', type=str, required=True, help="Target column name")
    parser.add_argument('--username', type=str, default="anonymous", help="Username of the teammate running the simulation")
    parser.add_argument('--task_type', type=str, choices=['classification', 'regression'], default='classification', help="Type of task")
    parser.add_argument('--distribution', type=str, choices=['iid', 'non_iid'], default='iid', help="Data distribution across clients")
    parser.add_argument('--num_clients', type=int, default=5, help="Number of FL clients")
    parser.add_argument('--num_rounds', type=int, default=10, help="Number of communication rounds")
    parser.add_argument('--local_epochs', type=int, default=5, help="Local epochs per client")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--proximal_mu', type=float, default=0.01, help="Proximal term for FedProx")
    
    args = parser.parse_args()
    filepath, err = execute_experiment(args)
    if err:
        print(f"Error: {err}")
    else:
        print(f"Simulation completed successfully: {filepath}")

if __name__ == "__main__":
    main()
