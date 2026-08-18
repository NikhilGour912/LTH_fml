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
import gc

config = {
    "learning_rate": 0.01,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "batch_size": 128,
    "epochs": 20,
    "model_name": "vgg16",
    "dataset": "cifar100",
    "transfer_source": "cifar10",
    "target_sparsity": 73.79, 
    "seed": 42
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

if not torch.backends.mps.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("mps")
print(f"Using device: {device}")

set_seed(config["seed"])

print("Loading CIFAR-100 data...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), 
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

def get_model(num_classes=100):
    set_seed(config["seed"])
    model = vgg16(weights=None, num_classes=num_classes).to(device)
    return model

def train(epoch, model, optimizer, criterion, mask=None):
    model.train()
    progress_bar = tqdm(trainloader, desc=f"Train Epoch {epoch}")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if mask:
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
    return acc

def get_cifar10_mask_oneshot(baseline_path, prune_percent):
    print(f"Generating mask from CIFAR-10 baseline: {baseline_path}")
    model_c10 = vgg16(weights=None, num_classes=10).to(device)
    model_c10.load_state_dict(torch.load(baseline_path, map_location=device))
    
    all_weights = []
    for name, param in model_c10.named_parameters():
        if 'weight' in name:
            all_weights.append(param.data.abs().view(-1))
    all_weights = torch.cat(all_weights)
    
    num_to_prune = int(len(all_weights) * prune_percent / 100.0)
    threshold = torch.kthvalue(all_weights, num_to_prune).values
    
    mask = {}
    for name, param in model_c10.named_parameters():
        if 'weight' in name:
            mask[name] = (param.data.abs() > threshold).float()
        else:
            mask[name] = torch.ones_like(param.data)
            
    del model_c10
    return mask

if __name__ == "__main__":
    
    BASELINE_C10_PATH = "baseline_vgg16_m4.pth"
    if not os.path.exists(BASELINE_C10_PATH):
        print("Error: CIFAR-10 Baseline not found. Run run_lth.py first.")
        exit()

    print("\n--- EXPERIMENT A: TRANSFERABILITY TO CIFAR-100 ---")
    
    print("\n>>> Running Part 1: CIFAR-100 Dense Baseline (Control)")
    
    wandb.init(project="lth-transfer-cifar100", name="cifar100-dense-baseline", config=config, reinit=True)
    
    model_dense = get_model(num_classes=100) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_dense.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_acc_dense = 0.0
    for epoch in range(config["epochs"]):
        train(epoch, model_dense, optimizer, criterion) 
        acc = test(epoch, model_dense, criterion)
        scheduler.step()
        wandb.log({"epoch": epoch, "test/acc": acc, "type": "dense"})
        if acc > best_acc_dense: best_acc_dense = acc
            
    print(f"** Best CIFAR-100 Dense Accuracy: {best_acc_dense:.2f}% **")
    wandb.finish()
    
    del model_dense
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    print("\n>>> Running Part 2: CIFAR-10 Ticket Transfer")
    
    mask_c10 = get_cifar10_mask_oneshot(BASELINE_C10_PATH, prune_percent=config["target_sparsity"])
    
    model_transfer = get_model(num_classes=100)
    
    print("Adapting mask for CIFAR-100...")
    final_mask = {}
    for name, m_tensor in mask_c10.items():
        if "classifier" in name:
            continue 
        final_mask[name] = m_tensor
    
    with torch.no_grad():
        for name, param in model_transfer.named_parameters():
            if name in final_mask:
                param.data *= final_mask[name].to(device)

    wandb.init(project="lth-transfer-cifar100", name="cifar100-transfer-ticket", config=config, reinit=True)
    
    optimizer = optim.SGD(model_transfer.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_acc_transfer = 0.0
    for epoch in range(config["epochs"]):
        train(epoch, model_transfer, optimizer, criterion, mask=final_mask) 
        acc = test(epoch, model_transfer, criterion)
        scheduler.step()
        wandb.log({"epoch": epoch, "test/acc": acc, "type": "transfer"})
        if acc > best_acc_transfer: best_acc_transfer = acc

    print(f"** Best CIFAR-100 Transfer Accuracy: {best_acc_transfer:.2f}% **")
    
    print("\n=== FINAL RESULTS ===")
    print(f"Dense CIFAR-100: {best_acc_dense:.2f}%")
    print(f"Transferred Ticket: {best_acc_transfer:.2f}%")
    
    wandb.finish()