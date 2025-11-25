import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import asha_metrics as am  # Your plotting functions
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = 150
global_rounds = 15
local_epochs = 12
batch_size = 32
learning_rate = 0.01
alpha, beta, gamma = 0.5, 0.15, 0.35

# Simple CNN for FashionMNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Input image = 28x28 -> after two 2x2 pools -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset
data_dir = './data'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
download = not os.path.exists(os.path.join(data_dir, 'FashionMNIST'))
train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform)

# Split data across clients
indices = list(range(len(train_dataset)))
random.shuffle(indices)
client_data_ratios = np.random.dirichlet(np.ones(num_clients), size=1)[0]
client_data_ratios = (client_data_ratios * len(train_dataset)).astype(int)
client_data_ratios[-1] = len(train_dataset) - client_data_ratios[:-1].sum()

client_datasets = []
start = 0
for samples in client_data_ratios:
    end = start + samples
    client_datasets.append(Subset(train_dataset, indices[start:end]))
    start = end

def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    total, correct = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            total += target.size(0)
        correct = sum(np.array(y_pred) == np.array(y_true))
    return y_true, y_pred, correct / total if total > 0 else 0

def train_client(model, dataset, lr, epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model

# Global test loader
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Tracking variables
fedavg_accuracies = []
asha_accuracies = []
per_round_fedavg_accuracies = {i: [] for i in range(num_clients)}
per_round_asha_accuracies = {i: [] for i in range(num_clients)}
client_contributions = {i: [] for i in range(num_clients)}
roundwise_latencies = {r: [random.uniform(10, 50) for _ in range(num_clients)] for r in range(global_rounds)}
dropout_frequency = {i: random.randint(0, 5) for i in range(num_clients)}
client_bandwidth = {i: random.uniform(10, 100) for i in range(num_clients)}
client_energy = {i: random.uniform(5, 20) for i in range(num_clients)}
client_final_accuracy_fedavg = {}
client_final_accuracy_asha = {}

fedavg_model = SimpleCNN().to(device)
asha_model = SimpleCNN().to(device)

# Federated Training
for round_num in range(global_rounds):
    fedavg_weights = []
    asha_weights = []
    acc_list, loss_list, res_list = [], [], []

    for cid in range(num_clients):
        local_model_fedavg = SimpleCNN().to(device)
        local_model_fedavg.load_state_dict(fedavg_model.state_dict())
        local_model_asha = SimpleCNN().to(device)
        local_model_asha.load_state_dict(asha_model.state_dict())

        client_dataloader = DataLoader(client_datasets[cid], batch_size=batch_size, shuffle=False)

        trained_model_fedavg = train_client(local_model_fedavg, client_datasets[cid], learning_rate, local_epochs)
        fedavg_weights.append({k: v.cpu() for k, v in trained_model_fedavg.state_dict().items()})

        trained_model_asha = train_client(local_model_asha, client_datasets[cid], learning_rate, local_epochs)
        y_true_local_asha, y_pred_local_asha, after_acc_asha = evaluate(trained_model_asha, client_dataloader)
        acc_list.append(after_acc_asha)
        loss_list.append(1 - after_acc_asha)
        res_list.append(random.uniform(0.5, 1.0))
        trained_weights_asha = {k: v.cpu() for k, v in trained_model_asha.state_dict().items()}
        asha_weights.append(trained_weights_asha)

    contributions_raw = [
        alpha * acc + beta * (1 - loss) + gamma * res
        for acc, loss, res in zip(acc_list, loss_list, res_list)
    ]
    total_score = sum(contributions_raw)
    normalized_contributions = [c / total_score for c in contributions_raw]
    for i, contrib in enumerate(normalized_contributions):
        client_contributions[i].append(contrib)

    # Aggregation - FedAvg
    new_weights_fedavg = fedavg_weights[0]
    for k in new_weights_fedavg:
        new_weights_fedavg[k] = sum([client[k] for client in fedavg_weights]) / num_clients
    fedavg_model.load_state_dict(new_weights_fedavg)

    # Aggregation - ASHA
    new_weights_asha = asha_weights[0]
    for k in new_weights_asha:
        new_weights_asha[k] = sum([
            client[k] * normalized_contributions[i] for i, client in enumerate(asha_weights)
        ])
    asha_model.load_state_dict(new_weights_asha)

    _, _, fedavg_acc = evaluate(fedavg_model, test_loader)
    _, _, asha_acc = evaluate(asha_model, test_loader)

    fedavg_accuracies.append(fedavg_acc)
    asha_accuracies.append(asha_acc)
    print(f"Round {round_num+1} - FedAvg Acc: {fedavg_acc:.4f} | ASHA Acc: {asha_acc:.4f}")

# --- Plotting Section ---
rounds = range(1, global_rounds + 1)
am.plot_accuracy_loss(fedavg_accuracies, [1 - acc for acc in fedavg_accuracies], rounds)
plt.title("FedAvg Accuracy and Loss (FashionMNIST)")
plt.show()

am.plot_accuracy_loss(asha_accuracies, [1 - acc for acc in asha_accuracies], rounds)
plt.title("ASHA Accuracy and Loss (FashionMNIST)")
plt.show()

am.plot_per_round_accuracy(per_round_fedavg_accuracies)
plt.title("Per-client Accuracy over Rounds - FedAvg (FashionMNIST)")
plt.show()

am.plot_per_round_accuracy(per_round_asha_accuracies)
plt.title("Per-client Accuracy over Rounds - ASHA (FashionMNIST)")
plt.show()

am.plot_client_contributions(client_contributions)
plt.title("Client Contribution per Round - ASHA (FashionMNIST)")
plt.show()

am.plot_roundwise_latency(roundwise_latencies)
plt.title("Round-wise Average Latency (FashionMNIST)")
plt.show()

am.plot_dropout_frequency(dropout_frequency)
plt.title("Client Dropout Frequency (FashionMNIST)")
plt.show()

# --- ROC Curve ---
class_labels = test_dataset.classes
n_classes = len(class_labels)

fedavg_y_true, fedavg_y_pred, _ = evaluate(fedavg_model, test_loader)
fedavg_y_true_binary = label_binarize(fedavg_y_true, classes=range(n_classes))
fedavg_y_score = []
with torch.no_grad():
    fedavg_model.eval()
    for data, target in test_loader:
        data = data.to(device)
        outputs = torch.softmax(fedavg_model(data), dim=1)
        fedavg_y_score.extend(outputs.cpu().numpy())
fedavg_y_score = np.array(fedavg_y_score)
am.plot_roc_curve(fedavg_y_true_binary, fedavg_y_score, n_classes)
plt.title("ROC Curve - FedAvg (FashionMNIST)")
plt.show()

asha_y_true, asha_y_pred, _ = evaluate(asha_model, test_loader)
asha_y_true_binary = label_binarize(asha_y_true, classes=range(n_classes))
asha_y_score = []
with torch.no_grad():
    asha_model.eval()
    for data, target in test_loader:
        data = data.to(device)
        outputs = torch.softmax(asha_model(data), dim=1)
        asha_y_score.extend(outputs.cpu().numpy())
asha_y_score = np.array(asha_y_score)
am.plot_roc_curve(asha_y_true_binary, asha_y_score, n_classes)
plt.title("ROC Curve - ASHA (FashionMNIST)")
plt.show()

# --- Precision-Recall-F1 ---
am.plot_precision_recall_f1(fedavg_y_true, fedavg_y_pred)
plt.title("Precision, Recall, F1 - FedAvg (FashionMNIST)")
plt.show()

am.plot_precision_recall_f1(asha_y_true, asha_y_pred)
plt.title("Precision, Recall, F1 - ASHA (FashionMNIST)")
plt.show()
