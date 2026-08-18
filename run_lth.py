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
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "or not supported hardware.")
    device = torch.device("cpu")
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


def get_model(save_initial=False):
    """Initializes the VGG-16 model."""
    set_seed(config["seed"]) 
    model = vgg16(weights=None, num_classes=10).to(device)
    
    if save_initial:
        torch.save(model.state_dict(), 'initial_weights_w0.pth')
        print("Saved initial weights (w_0) to 'initial_weights_w0.pth'")
        
    return model

def train(epoch, model, optimizer, criterion):
    """Standard training loop for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc=f"Train Epoch {epoch}")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar.set_postfix(loss=train_loss/(batch_idx+1), acc=100.*correct/total)
    
    wandb.log({"train/epoch_loss": train_loss/(batch_idx+1), "train/epoch_acc": 100.*correct/total, "epoch": epoch})

def test(epoch, model, criterion):
    """Standard testing loop for one epoch."""
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
            progress_bar.set_postfix(loss=test_loss/(batch_idx+1), acc=100.*correct/total)

    acc = 100.*correct/total
    wandb.log({"test/epoch_loss": test_loss/(batch_idx+1), "test/epoch_acc": acc, "epoch": epoch})
    return acc

def train_sparse(epoch, model, optimizer, criterion, mask):
    """Training loop that re-applies the mask after every step."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
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

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar.set_postfix(loss=train_loss/(batch_idx+1), acc=100.*correct/total)
    
    wandb.log({"train/epoch_loss": train_loss/(batch_idx+1), "train/epoch_acc": 100.*correct/total, "epoch": epoch})

def get_pruning_mask(model, prune_percent):
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
        else:
            mask[name] = torch.ones_like(param.data)
    return mask


if __name__ == "__main__":
    os.makedirs('./data', exist_ok=True)
    
    BASELINE_MODEL_PATH = "baseline_vgg16_m4.pth"
    INITIAL_WEIGHTS_PATH = "initial_weights_w0.pth"
    MASK_PATH = f"mask_p{config['prune_percent']}_m4.pth"
    
    best_baseline_acc = 0.0
    best_sparse_acc = 0.0

    print("\n--- PART 1: TRAINING DENSE BASELINE ---")
    
    wandb.init(
        project="lth-investigation-m4",
        name="1-baseline-dense",
        config=config,
        reinit=True
    )
    
    model = get_model(save_initial=True) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    for epoch in range(config["epochs"]):
        train(epoch, model, optimizer, criterion)
        acc = test(epoch, model, criterion)
        scheduler.step()
        
        if acc > best_baseline_acc:
            best_baseline_acc = acc
            print(f"New best baseline accuracy: {acc:.2f}%. Saving model...")
            torch.save(model.state_dict(), BASELINE_MODEL_PATH)
            
    print(f"--- Finished Baseline Training ---")
    print(f"** Best M4 Baseline Accuracy: {best_baseline_acc:.2f}% **")
    wandb.finish()

    print("\n--- PART 2: GENERATING PRUNING MASK ---")
    
    model = get_model() 
    model.load_state_dict(torch.load(BASELINE_MODEL_PATH))
    print(f"Loaded trained baseline model from {BASELINE_MODEL_PATH}")
    
    pruning_mask = get_pruning_mask(model, config["prune_percent"])
    torch.save(pruning_mask, MASK_PATH)
    print(f"Generated and saved mask to {MASK_PATH}")

    print("\n--- PART 3: TRAINING WINNING TICKET ---")
    
    config["architecture"] = f"VGG-16 Sparse Ticket (p={config['prune_percent']}%)"
    wandb.init(
        project="lth-investigation-m4",
        name=f"2-winning-ticket-p{config['prune_percent']}",
        config=config,
        reinit=True
    )
    
    ticket_model = get_model()
    
    mask = torch.load(MASK_PATH)
    
    with torch.no_grad():
        for name, param in ticket_model.named_parameters():
            if name in mask:
                param.data *= mask[name].to(device)
    print("Loaded initial weights (w_0) and applied mask (m).")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ticket_model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    for epoch in range(config["epochs"]):
        train_sparse(epoch, ticket_model, optimizer, criterion, mask)
        acc = test(epoch, ticket_model, criterion)
        scheduler.step()
        
        if acc > best_sparse_acc:
            best_sparse_acc = acc
            print(f"New best sparse accuracy: {acc:.2f}%.")
            
    print(f"--- Finished Sparse Ticket Training ---")
    print(f"\n--- FINAL RESULTS (M4) ---")
    print(f"** Best Baseline (Dense) Accuracy: {best_baseline_acc:.2f}% **")
    print(f"** Best Ticket (Sparse) Accuracy: {best_sparse_acc:.2f}% **")
    wandb.finish()