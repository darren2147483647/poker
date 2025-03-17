from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
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

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, gt, test_size=0.2, random_state=42)

# 建立 AdaBoost 模型，基礎學習器使用決策樹
model = AdaBoostClassifier(n_estimators=50, random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估準確率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 計算指標
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


# 計算混淆矩陣
cm = confusion_matrix(y_test, y_pred)

# 繪製混淆矩陣
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

