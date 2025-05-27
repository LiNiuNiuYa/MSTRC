import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import random


def set_seed(seed_value=42):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MSTRC(nn.Module):
    def __init__(self,
                 num_features_xd=43,  # 分子图节点特征数
                 hidden_dim=128,  # TransformerConv 输出维度
                 heads=4,  # Multi-head 注意力的 head 数
                 dropout=0.3
                 ):

        super(MSTRC, self).__init__()

        # TransformerConv层
        self.trans_conv1 = TransformerConv(
            in_channels=num_features_xd,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout
        )
        # 线性层，将多头注意力的输出转换回hidden_dim维度
        self.lin_mol_1 = nn.Linear(hidden_dim * heads, hidden_dim)

        self.trans_conv2 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout
        )
        self.lin_mol_2 = nn.Linear(hidden_dim * heads, hidden_dim)


        self.trans_conv3 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout
        )
        self.lin_mol_3 = nn.Linear(hidden_dim * heads, hidden_dim)

        # 全连接层
        self.fc_g = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
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

        # 第一层 TransformerConv
        x_g = self.trans_conv1(x, edge_index)
        x_g = self.lin_mol_1(x_g)  # 将维度转回 hidden_dim
        x_g = F.relu(x_g)

        # 第二层 TransformerConv
        x_g = self.trans_conv2(x_g, edge_index)
        x_g = self.lin_mol_2(x_g)
        x_g = F.relu(x_g)

        # 第三层 TransformerConv
        x_g = self.trans_conv3(x_g, edge_index)
        x_g = self.lin_mol_3(x_g)
        x_g = F.relu(x_g)

        # 池化（max + mean）
        x_g = torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)

        # MLP处理
        x_g = self.fc_g(x_g)

        # 最终输出
        z = self.fc_final(x_g)

        return z