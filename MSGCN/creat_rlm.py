from sklearn.utils import resample
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
import os
# 导入分子处理函数
from bondedgeconstruction.data.featurization import smiles_to_3d_mol
from bondedgeconstruction.data.featurization import mol_to_data


def smiles_to_molecular_graph(smiles):

    mol = smiles_to_3d_mol(smiles)
    if mol is None:
        return None

    data = mol_to_data(mol)
    return data


# 固定随机种子
RANDOM_SEED = 42

# 数据路径
root = 'data_rlm2'
data_rlm = pd.read_csv(root + '/test.csv')

# 平衡数据集
stable = data_rlm[data_rlm['Label'] == 1]  # 稳定化合物
unstable = data_rlm[data_rlm['Label'] == 0]  # 不稳定化合物

# 随机采样不稳定化合物，数量与稳定化合物一致
unstable_sampled = resample(
    unstable,
    n_samples=len(stable),
    random_state=RANDOM_SEED,
    replace=False
)

# 合并平衡数据
balanced_data = pd.concat([stable, unstable_sampled])

# 划分训练集和测试集（80% 训练，20% 测试）
train_data = balanced_data.sample(frac=0.8, random_state=RANDOM_SEED)
test_data = balanced_data.drop(train_data.index)

# 保存平衡数据
train_data.to_csv(root + '/train_balanced.csv', index=False)
test_data.to_csv(root + '/test_balanced.csv', index=False)


if not os.path.exists(root + '/train.pth'):
    dataset_train = []
    for _, row in tqdm(train_data.iterrows()):
        data = smiles_to_molecular_graph(row['SMILES'])
        if data is not None:
            data.y = torch.tensor(row['Label'], dtype=torch.long)
            dataset_train.append(data)
    torch.save(dataset_train, root + '/train.pth')

if not os.path.exists(root + '/test.pth'):
    dataset_test = []
    for _, row in tqdm(test_data.iterrows()):
        data = smiles_to_molecular_graph(row['SMILES'])
        if data is not None:
            data.y = torch.tensor(row['Label'], dtype=torch.long)
            dataset_test.append(data)
    torch.save(dataset_test, root + '/test.pth')