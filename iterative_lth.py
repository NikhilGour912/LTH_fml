import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16
from tqdm import tqdm
import wandb
import numpy as np
import random
import os
import copy

config = {
    "learning_rate": 0.01,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "batch_size": 128,
    "epochs": 100,
    "model_name": "vgg16",
    "dataset": "cifar10",
    "prune_percent": 20,
    "prune_iterations": 10,
    "seed": 42
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

if not torch.backends.mps.is_available():
    device = torch.device("cpu")
    print("MPS not available. Using CPU.")
else:
    device = torch.device("mps")
print(f"Using device: {device}")
print(f"Config: {config}")

set_seed(config["seed"])

print("Loading CIFAR-10 data...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

def get_model():
    set_seed(config["seed"])
    model = vgg16(weights=None, num_classes=10).to(device)
    return model

def train_sparse(epoch, model, optimizer, criterion, mask):
    model.train()
    progress_bar = tqdm(trainloader, desc=f"Sparse Train Epoch {epoch}")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask:
                    param.data *= mask[name].to(device)

def test(epoch, model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc=f"Test Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100.*correct/total
    avg_loss = test_loss / (batch_idx + 1)
    return acc, avg_loss

def get_pruning_mask(model, prune_percent, existing_mask):
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            alive_weights = param.data.abs()[existing_mask[name] == 1]
            all_weights.append(alive_weights.view(-1))
    
    all_weights = torch.cat(all_weights)
    
    num_to_prune = int(len(all_weights) * prune_percent / 100.0)
    
    if num_to_prune == 0:
        print("No weights to prune. Returning original mask.")
        return existing_mask

    threshold = torch.kthvalue(all_weights, num_to_prune).values
    
    new_mask = {}
    for name, param in model.named_parameters():
        if name not in existing_mask:
            new_mask[name] = torch.ones_like(param.data)
            continue
        
        new_mask[name] = existing_mask[name].clone().to(device)
        
        if 'weight' in name:
            weights_to_prune = (param.data.abs() < threshold) & (existing_mask[name] == 1)
            new_mask[name][weights_to_prune] = 0.0
            
    return new_mask

def apply_mask_to_model(model, mask):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name].to(device)
                
def get_initial_mask(model):
    mask = {}
    for name, param in model.named_parameters():
        mask[name] = torch.ones_like(param.data).to(device)
    return mask

def get_sparsity(mask):
    total_weights = 0
    total_pruned = 0
    for name, tensor in mask.items():
        if 'weight' in name:
            total_weights += tensor.numel()
            total_pruned += (tensor == 0).sum().item()
    
    if total_weights == 0:
        return 0.0
        
    return 100.0 * total_pruned / total_weights

if __name__ == "__main__":
    
    BASELINE_MODEL_PATH = "baseline_vgg16_m4.pth"
    INITIAL_WEIGHTS_PATH = "initial_weights_w0.pth"
    
    if not os.path.exists(BASELINE_MODEL_PATH) or not os.path.exists(INITIAL_WEIGHTS_PATH):
        print(f"Baseline '{BASELINE_MODEL_PATH}' or initial weights '{INITIAL_WEIGHTS_PATH}' not found.")
        print("Please run 'run_lth.py' first to generate these files.")
        exit()
        
    print("--- STARTING ITERATIVE MAGNITUDE PRUNING (IMP) ---")
    
    initial_weights = torch.load(INITIAL_WEIGHTS_PATH, map_location=device)
    print(f"Loaded initial weights (w_0) from {INITIAL_WEIGHTS_PATH}")

    trained_model = get_model()
    trained_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device))
    print(f"Loaded trained baseline model from {BASELINE_MODEL_PATH}")

    cumulative_mask = get_initial_mask(trained_model)
    
    for i in range(config["prune_iterations"]):
        
        current_sparsity = get_sparsity(cumulative_mask)
        print(f"\n--- STARTING IMP ITERATION {i+1}/{config['prune_iterations']} ---")
        print(f"Current Sparsity: {current_sparsity:.2f}%")

        print(f"Pruning {config['prune_percent']}% of remaining weights...")
        cumulative_mask = get_pruning_mask(
            trained_model, 
            config["prune_percent"], 
            cumulative_mask
        )
        
        new_sparsity = get_sparsity(cumulative_mask)
        print(f"New Sparsity: {new_sparsity:.2f}%")
        
        ticket_model = get_model()
        ticket_model.load_state_dict(initial_weights)
        apply_mask_to_model(ticket_model, cumulative_mask)
        print("Model rewound to w_0 and new mask applied.")

        wandb.init(
            project="lth-imp-m4",
            name=f"imp_iter_{i+1}_sparsity_{new_sparsity:.2f}",
            config=config,
            reinit=True
        )
        wandb.config.update({"iteration": i+1, "sparsity": new_sparsity})

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(ticket_model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        
        best_acc_this_iter = 0.0
        
        for epoch in range(config["epochs"]):
            train_sparse(epoch, ticket_model, optimizer, criterion, cumulative_mask)
            acc, loss = test(epoch, ticket_model, criterion)
            
            wandb.log({
                "epoch": epoch,
                "test/epoch_acc": acc,
                "test/epoch_loss": loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            if acc > best_acc_this_iter:
                best_acc_this_iter = acc
                
            scheduler.step()
            
        print(f"--- Iteration {i+1} Finished ---")
        print(f"** Best Accuracy (Sparsity {new_sparsity:.2f}%): {best_acc_this_iter:.2f}% **")
        wandb.log({"final_best_accuracy": best_acc_this_iter})
        wandb.finish()
        
        trained_model = ticket_model

    print("\n--- ALL IMP ITERATIONS COMPLETE ---")