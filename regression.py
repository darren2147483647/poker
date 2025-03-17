import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error
import label_extract

class PokerDataset(Dataset):
    def __init__(self, folder_path):
        self.data = []
        self.targets = []
        self.phh_info, self.single_info, self.single_key = label_extract.extract_from_folder(folder_path)
        self.preprocess_data()

    def preprocess_data(self):
        # 單一人 前注+盲注+起始籌碼+手牌1+手牌2+位置->預測最終籌碼regression
        # self.single_key = ['antes', 'blinds_or_straddles', 'starting_stacks', 'deck1', 'deck2', 'seat', 'play', 'finishing_stacks', 'win_or_lose_chip_amount', 'win_or_lose_chip_condition', 'bankrupt']
        data_interest = ['antes', 'blinds_or_straddles', 'starting_stacks', 'deck1', 'deck2', 'seat']
        target_interest = ['win_or_lose_chip_amount',]
        self.data = [[x[key] for key in data_interest] for x in self.phh_info]
        self.targets = [[x[key] for key in target_interest] for x in self.phh_info]
        return

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.targets[idx], dtype=torch.float)

class PokerNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64]):
        super(PokerNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x.view(x.size(0), 1, -1)

def train_model(dataloader, input_size, output_size):
    model = PokerNet(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        for features, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    return model

def inference(model, input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float).detach()
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)
        output = model(input_tensor)
    model.train()
    return output

def visualize(predict, targets):
    """
    可視化每一批次的預測值與真實目標值，將所有批次的數據拼接在一起。
    
    參數：
    - predict: 模型的預測結果，形狀為 (B, N)
    - targets: 真實目標值，形狀為 (B, N)
    """
    if isinstance(predict,torch.Tensor):
        predict_np = predict.numpy()
        targets_np = targets.numpy()
    else:
        predict_np = predict
        targets_np = targets
    N = predict_np.shape[1] #num of players
    player_labels = [f"Seat {i+1}" for i in range(N)]
    plt.figure(figsize=(10, 6))
    for i in range(predict_np.shape[0]):
        plt.plot(player_labels, predict_np[i], marker='o', linestyle='-', label=f"Batch {i+1} Prediction", color='red')
        plt.plot(player_labels, targets_np[i], marker='s', linestyle='--', label=f"Batch {i+1} Target", color='green')

    plt.title("Prediction vs Target")
    plt.text(0.5, 1.05, "red:prediction green:GT", ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.ylabel("chip +-")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# 設定資料夾路徑
folder_path1 = "train"  # 替換成訓練資料夾路徑
dataset1 = PokerDataset(folder_path1)
dataloader1 = DataLoader(dataset1, batch_size=4, shuffle=True)

folder_path2 = "test"  # 替換成測試資料夾路徑
dataset2 = PokerDataset(folder_path2)
dataloader2 = DataLoader(dataset2, batch_size=4, shuffle=False)

C_in, N = (dataset1[0][0].shape)
C_out, N = (dataset1[0][1].shape)
input_size = C_in * N
output_size = C_out * N
model = train_model(dataloader1, input_size, output_size)
print("Training completed.")

all_predict = []
all_targets = []
for features, targets in dataloader2:
    predict = inference(model, features)
    all_predict.append(predict)
    all_targets.append(targets)
all_predict = np.concatenate([p.squeeze(dim = 1).numpy() for p in all_predict], axis=0)
all_targets = np.concatenate([t.squeeze(dim = 1).numpy() for t in all_targets], axis=0)
error = (all_predict - all_targets)
mean_error = np.mean(np.abs(error))
mse_error = np.mean((error)*(error))
print(f"average chips eror: {mean_error}")
print(f"mse: {mse_error}")
visualize(all_predict, all_targets)


### k-fold
# 設定 K-Fold 交叉驗證的折數
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 使用所有數據
folder_path = "hand_record"  # 替換成訓練資料夾路徑
dataset = PokerDataset(folder_path)

# 轉換數據格式
features = np.array(dataset.data)
targets = np.array(dataset.targets)

mse_scores = []
mae_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
    print(f"Fold {fold + 1}/{k}")

    # 切分訓練集與驗證集
    X_train, X_val = features[train_idx], features[val_idx]
    y_train, y_val = targets[train_idx], targets[val_idx]

    # 轉換為 PyTorch 張量
    train_dataset = list(zip(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float)))
    val_dataset = list(zip(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 訓練模型
    model = train_model(train_loader, input_size, output_size)

    # 在驗證集上進行預測
    all_predict = []
    all_targets = []
    for features, targets in val_loader:
        predict = inference(model, features)
        all_predict.append(predict.numpy())
        all_targets.append(targets.numpy())

    # 合併所有批次的預測結果
    all_predict = np.concatenate([p.squeeze(1) for p in all_predict], axis=0)
    all_targets = np.concatenate([t.squeeze(1) for t in all_targets], axis=0)

    # 計算 MSE 和 MAE
    mse = mean_squared_error(all_targets, all_predict)
    mae = mean_absolute_error(all_targets, all_predict)
    mse_scores.append(mse)
    mae_scores.append(mae)

    print(f"Fold {fold + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}")

# 計算平均誤差
mean_mse = np.mean(mse_scores)
mean_mae = np.mean(mae_scores)

print(f"\nFinal Cross-Validation Results:")
print(f"Average MSE: {mean_mse:.4f}")
print(f"Average MAE: {mean_mae:.4f}")

