import os
import sys

sys.path.append(".")
# 忽略烦人的红色提示
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from pytorch_metric_learning import losses

warnings.filterwarnings("ignore")

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device", device)

train_transform = transforms.Compose(
    [
        transforms.Resize((800, 800)),
        transforms.RandomAdjustSharpness(5.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose(
    [
        transforms.Resize((800, 800)),
        transforms.RandomAdjustSharpness(5.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

dataset_dir = "MyDataset_split"
train_path = os.path.join(dataset_dir, "train")
test_path = os.path.join(dataset_dir, "val")
print("训练集路径", train_path)
print("测试集路径", test_path)


# # 载入训练集
train_dataset = datasets.ImageFolder(train_path, train_transform)
# # 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)


print("训练集图像数量", len(train_dataset))
print("类别个数", len(train_dataset.classes))
print("各类别名称", train_dataset.classes)
print("测试集图像数量", len(test_dataset))
print("类别个数", len(test_dataset.classes))
print("各类别名称", test_dataset.classes)

# 各类别名称
class_names = train_dataset.classes
n_class = len(class_names)
# 映射关系：类别 到 索引号
train_dataset.class_to_idx
# 映射关系：索引号 到 类别
idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
# ------=========================================----


BATCH_SIZE = 32
# 训练集的数据加载器
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

# 测试集的数据加载器
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)


class MyModel(nn.Module):
    def __init__(self, n_class):
        super(MyModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.act = nn.ReLU()
        self.FC = nn.Linear(self.model.fc.out_features, n_class)

        # 交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.loss_func1 = losses.ProxyAnchorLoss(n_class, self.model.fc.out_features)
        self.loss_func2 = losses.TripletMarginLoss()

    def forward(self, x, y=None):
        out = self.model(x)

        logits = self.FC(self.act(out))

        if y is not None:
            loss1 = self.criterion(logits, y)
            loss2 = self.loss_func1(out, y)
            loss3 = self.loss_func2(out, y)
            loss = 0.7 * loss1 + 0.1 * loss2 + 0.2 * loss3
            return logits, loss
        else:
            return logits


model = MyModel(n_class)  # 载入预训练模型

optimizer = optim.Adam(model.parameters(), lr=5e-5)

model = model.to(device)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()


# 学习率降低策略
lr_schedulerM = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def train_epoch():
    """
    运行一个 batch 的训练，返回当前 batch 的训练日志
    """
    ## 训练阶段
    labels_list = []
    preds_list = []
    loss_list = []

    model.train()
    for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
        # 获得一个 batch 的数据和标注
        images = images.to(device)
        labels = labels.to(device)

        logits, loss = model(images, labels)  # 输入模型，执行前向预测

        # 优化更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 获取当前 batch 的标签类别和预测类别
        _, preds = torch.max(logits, 1)  # 获得当前 batch 所有图像的预测类别
        preds = preds.cpu().numpy()
        loss = loss.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        labels_list.extend(labels)
        preds_list.extend(preds)
        loss_list.append(loss)

    log_train = {}
    log_train["epoch"] = epoch
    log_train["batch"] = batch_idx
    # 计算分类评估指标
    log_train["train_loss"] = sum(loss_list) / len(loss_list)
    log_train["train_precision"] = precision_score(
        labels_list, preds_list, average="macro"
    )
    log_train["train_recall"] = recall_score(labels_list, preds_list, average="macro")
    log_train["train_f1-score"] = f1_score(labels_list, preds_list, average="macro")

    print("log_train ==>", log_train)
    return log_train


def evaluate_epoch():
    """
    在整个测试集上评估，返回分类评估指标日志
    """
    labels_list = []
    preds_list = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            labels = labels.detach().cpu().numpy()
            labels_list.extend(labels)
            preds_list.extend(preds)

    log_test = {}
    log_test["epoch"] = epoch

    # 计算分类评估指标
    log_test["test_accuracy"] = accuracy_score(labels_list, preds_list)
    log_test["test_precision"] = precision_score(
        labels_list, preds_list, average="macro"
    )
    log_test["test_recall"] = recall_score(labels_list, preds_list, average="macro")
    log_test["test_f1-score"] = f1_score(labels_list, preds_list, average="macro")
    print("log_test ==>", log_test)
    return log_test


# 训练轮次 Epoch
EPOCHS = 30
batch_idx = 0
best_test_accuracy = 0
df_train_log, df_test_log = [], []

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")

    ## 训练阶段
    log_train = train_epoch()
    df_train_log.append(log_train)
    lr_schedulerM.step()

    ## 测试阶段
    log_test = evaluate_epoch()
    df_test_log.append(log_test)

    # 保存最新的最佳模型文件
    if log_test["test_accuracy"] > best_test_accuracy:
        # 删除旧的最佳模型文件(如有)
        old_best_checkpoint_path = "checkpoints/best-{:.3f}.pth".format(
            best_test_accuracy
        )
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)
        # 保存新的最佳模型文件
        new_best_checkpoint_path = "checkpoints/best-{:.3f}.pth".format(
            log_test["test_accuracy"]
        )
        torch.save(model, new_best_checkpoint_path)
        print(
            "保存新的最佳模型", "checkpoints/best-{:.3f}.pth".format(best_test_accuracy)
        )
        best_test_accuracy = log_test["test_accuracy"]


df_train_log = pd.DataFrame(df_train_log)
df_train_log.to_csv("训练日志-训练集.csv", index=False)
df_test_log = pd.DataFrame(df_test_log)
df_test_log.to_csv("训练日志-测试集.csv", index=False)
