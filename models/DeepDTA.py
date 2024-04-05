import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool

class DeepDTA(nn.Module):
    def __init__(self,protein_MAX_LENGH = 1000, protein_kernel = [4,8,12],
                 drug_MAX_LENGH = 100, drug_kernel = [4,6,8],
                 conv = 32, char_dim = 128):
        super(DeepDTA, self).__init__()
        self.dim = char_dim
        self.conv = conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = protein_kernel

        self.protein_embed = nn.Embedding(26, self.dim,padding_idx=0 )
        self.drug_embed = nn.Embedding(65, self.dim,padding_idx=0 )
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels= self.conv,  kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels= self.conv*2,  kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels= self.conv*3,  kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.drug_MAX_LENGH-self.drug_kernel[0]-self.drug_kernel[1]-self.drug_kernel[2]+3)
        )
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        )
        self.fc1 = nn.Linear(192, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        # 修改最后一层输出
        self.out = nn.Linear(512, 2)    # self.out = nn.Linear(512, 1)
        # 添加激活函数
        self.softmax = nn.Softmax(dim=1)
        torch.nn.init.constant_(self.out.bias, 5)

    def forward(self, drug, protein):
        drugembed = self.drug_embed(drug)
        drugembed = drugembed.permute(0, 2, 1)
        drugConv = self.Drug_CNNs(drugembed)

        proteinembed = self.protein_embed(protein)
        proteinembed = proteinembed.permute(0, 2, 1)
        proteinConv = self.Protein_CNNs(proteinembed)

        pair = torch.cat([drugConv,proteinConv], dim=1)
        pair = torch.squeeze(pair, 2)
        
        fully1 = F.dropout(F.relu(self.fc1(pair)),p=0.1)
        fully2 = F.dropout(F.relu(self.fc2(fully1)),p=0.1)
        fully3 = F.relu(self.fc3(fully2))
        # 添加激活函数
        y = self.out(fully3)
        predict = self.softmax(y)

        # 输出原始预测分数和激活后的分数
        return predict, y


def remove_last_layer(model):
    new_model = DeepDTA().cuda()
    new_model.load_state_dict(model.state_dict())

    # nn.Identity表示输入什么就输出什么，即去除掉softmax层的影响，得到logits
    new_model.softmax = nn.Identity()

    return new_model