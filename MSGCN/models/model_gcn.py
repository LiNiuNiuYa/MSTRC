

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
import torch.nn as nn
import random


def set_seed(seed_value=42):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MSGCN(nn.Module):
    def __init__(self, num_features_xd=43, dropout=0.3):
        super(MSGCN, self).__init__()

        # GCN层
        self.conv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.conv3 = GCNConv(num_features_xd * 4, num_features_xd * 10)

        # 全连接层
        self.fc_g = nn.Sequential(
            nn.Linear(num_features_xd * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出层
        self.fc_final = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x, edge_index, batch):
        # 分子图卷积
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # 图池化
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # 全连接层
        x = self.fc_g(x)

        # 输出预测
        out = self.fc_final(x)

        return out
