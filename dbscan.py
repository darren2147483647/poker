import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import label_extract


class PokerDataset():
    def __init__(self, folder_path):
        self.data = []
        self.targets = []
        self.phh_info, self.single_info, self.single_key = label_extract.extract_from_folder(folder_path)
        self.preprocess_data()

    def preprocess_data(self):
        # 單一人 前注+盲注+起始籌碼+手牌1+手牌2+位置->預測籌碼增減狀況classify
        # self.single_key = ['antes', 'blinds_or_straddles', 'starting_stacks', 'deck1', 'deck2', 'seat', 'play', 'finishing_stacks', 'win_or_lose_chip_amount', 'win_or_lose_chip_condition', 'bankrupt']
        data_interest = ['antes', 'blinds_or_straddles', 'starting_stacks', 'deck1', 'deck2', 'seat']
        target_interest = ['win_or_lose_chip_condition',]
        self.data = [x[:6] for x in self.single_info]
        self.targets = [x[9] for x in self.single_info]
        return

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.data[idx]), (self.targets[idx])

# 設定資料夾路徑
folder_path = "hand_record"
dataset = PokerDataset(folder_path)

# 取得所有特徵資料
features = np.array(dataset.data)
gt = np.array(dataset.targets)

# 先標準化原始數據（在降維之前）
scaler = StandardScaler()
features_norm = scaler.fit_transform(features)

# 使用 PCA 將特徵降到 2 維
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_norm)

# 使用 DBSCAN 進行聚類分析
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(features_2d)

# 繪製 DBSCAN 結果
plt.figure(figsize=(5, 5))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
condition = ["L", "F", "W"]
for i, (x, y) in enumerate(features_2d):
    plt.text(x, y, condition[int(gt[i] + 1)], fontsize=8, ha='right')
plt.title("DBSCAN Clustering (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.show()

# print(features.shape, features_2d.shape)