from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import label_extract

class PokerDataset():
    def __init__(self, folder_path):
        self.data = []
        self.targets = []
        self.phh_info, self.single_info, self.single_key = label_extract.extract_from_folder(folder_path)
        self.preprocess_data()

    def preprocess_data(self):
        # 單一人 前注+盲注+起始籌碼+手牌1+手牌2+位置->預測入場classify
        # self.single_key = ['antes', 'blinds_or_straddles', 'starting_stacks', 'deck1', 'deck2', 'seat', 'play', 'finishing_stacks', 'win_or_lose_chip_amount', 'win_or_lose_chip_condition', 'bankrupt']
        data_interest = ['antes', 'blinds_or_straddles', 'starting_stacks', 'deck1', 'deck2', 'seat']
        target_interest = ['play',]
        self.data = [x[:6] for x in self.single_info]
        self.targets = [x[6] for x in self.single_info]
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

# 生成示例數據
X = np.random.rand(100, 5)  # 100筆樣本，每筆5個特徵
y = np.random.randint(0, 2, 100)  # 0 或 1 兩類

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, gt, test_size=0.2, random_state=42)

# 建立 AdaBoost 模型，基礎學習器使用決策樹
base_learner = DecisionTreeClassifier(max_depth=1)  # 弱學習器
model = AdaBoostClassifier(n_estimators=50, random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估準確率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")