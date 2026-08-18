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
    "learning_rate": 0.001, 
    "optimizer": "adamw",
    "batch_size": 128,
    "epochs": 100, # Speed run
    "model_name": "vgg16",
    "dataset": "cifar10",
    "prune_percent": 20, 
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

def train(epoch, model, optimizer, criterion):
    model.train()
    progress_bar = tqdm(trainloader, desc=f"Train Epoch {epoch}")
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

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

def generate_mask(model, prune_percent):
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.data.abs().view(-1))
    all_weights = torch.cat(all_weights)
    num_to_prune = int(len(all_weights) * prune_percent / 100.0)
    threshold = torch.kthvalue(all_weights, num_to_prune).values
    
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask[name] = (param.data.abs() > threshold).float()
    return mask

def calculate_stats(mask1, mask2):
    intersection = 0
    union = 0
    total_weights_pruned = 0
    
    for name in mask1:
        if name in mask2:
            m1 = mask1[name].byte()
            m2 = mask2[name].byte()
            
            intersection += (m1 & m2).sum().item()
            union += (m1 | m2).sum().item()
            
    iou = intersection / union if union > 0 else 0
    return iou

if __name__ == "__main__":
    
    SGD_BASELINE_PATH = "baseline_vgg16_m4.pth"
    if not os.path.exists(SGD_BASELINE_PATH):
        print("Error: SGD Baseline not found.")
        exit()

    print("\n--- EXPERIMENT B: OPTIMIZER STABILITY (SGD vs ADAMW) ---")
    print("\n>>> Step 1: Training Dense Model with AdamW")
    wandb.init(project="lth-optimizer-compare", name="adamw-dense-baseline", config=config, reinit=True)
    
    model_adam = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_adam.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    for epoch in range(config["epochs"]):
        train(epoch, model_adam, optimizer, criterion)
        acc = test(epoch, model_adam, criterion)
        scheduler.step()
        wandb.log({"epoch": epoch, "test/acc": acc})
    
    print(f"AdamW Baseline Accuracy: {acc:.2f}%")
    wandb.finish()


    print("\n>>> Step 2: Generating AdamW Mask")
    mask_adam = generate_mask(model_adam, config["prune_percent"])
    del model_adam
    gc.collect()

    print("\n>>> Step 3: Loading SGD Baseline & Generating Mask")
    model_sgd = get_model()
    model_sgd.load_state_dict(torch.load(SGD_BASELINE_PATH, map_location=device))
    mask_sgd = generate_mask(model_sgd, config["prune_percent"])
    del model_sgd
    gc.collect()
    print("\n>>> Step 4: Comparing Masks & Logging Graph")
    similarity = calculate_stats(mask_adam, mask_sgd)
    
    wandb.init(project="lth-optimizer-compare", name="final-analysis-graph", config=config, reinit=True)
    
    print("="*40)
    print(f"Jaccard Similarity (IoU): {similarity:.4f}")
    print("="*40)

    wandb.log({"jaccard_similarity": similarity})
    
    data = [
        ["Match (Intersection)", similarity],
        ["Mismatch (Difference)", 1.0 - similarity]
    ]
    table = wandb.Table(data=data, columns=["Category", "Ratio"])
    wandb.log({
        "optimizer_comparison_chart": wandb.plot.bar(
            table, "Category", "Ratio", title="SGD vs AdamW Mask Overlap"
        )
    })
    
    wandb.finish()
    print("Graph logged to WandB!")