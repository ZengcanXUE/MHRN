import torch
import numpy as np
import math
import torch.nn
from torch.nn import functional as F
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M


class DistMult(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(DistMult, self).__init__()

        self.emb_e = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_e.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)

        return pred


class ComplEx(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(ComplEx, self).__init__()

        self.emb_e_real = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_e_real.weight.data)
        torch.nn.init.xavier_normal_(self.emb_e_img.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class ConvE(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(ConvE, self).__init__()
        self.ent_dim = 200
        self.rel_dim = 200
        self.emb_e = torch.nn.Embedding(data.entities_num, self.ent_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(data.relations_num, self.rel_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = 20
        self.emb_dim2 = self.ent_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.ent_dim)
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(data.entities_num)))
        self.fc = torch.nn.Linear(9728, self.ent_dim)

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_e.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


class MHRN(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(MHRN, self).__init__()
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.reshape_H = 64
        self.reshape_W = 243
        self.out_1 = 8
        self.out_2 = 20
        self.out_3 = 8

        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_height = kwargs["filt_height"]
        self.filt_width = kwargs["filt_width"]

        self.entity_embedding = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.relation_embedding = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        filter1_dim = self.in_channels * self.out_1 * 3 * 3
        self.filter1 = torch.nn.Embedding(data.relations_num, filter1_dim, padding_idx=0)
        filter2_dim = self.in_channels * self.out_2 * 3 * 3
        self.filter3 = torch.nn.Embedding(data.relations_num, filter2_dim, padding_idx=0)
        filter3_dim = self.in_channels * self.out_3 * 1 * 9
        self.filter5 = torch.nn.Embedding(data.relations_num, filter3_dim, padding_idx=0)

        # Group equivariant convolution is associated with the group p4
        self.group_equivariant_conv1 = P4ConvZ2(self.in_channels, self.out_channels, 3, 1, 0)
        self.conv2 = P4MConvZ2(self.in_channels, self.out_channels, 3, 1, 0)

        self.input_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_1 + self.out_2 + self.out_3)

        self.bn1_1 = torch.nn.BatchNorm2d(self.out_1)
        self.bn1_2 = torch.nn.BatchNorm2d(self.out_2)
        self.bn1_3 = torch.nn.BatchNorm2d(self.out_3)
        self.bn2 = torch.nn.BatchNorm1d(ent_dim)

        self.loss = torch.nn.BCELoss()
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(data.entities_num)))

        fc_length = self.reshape_H * self.reshape_W * (self.out_1 + self.out_2 + self.out_3)
        self.fc = torch.nn.Linear(fc_length, ent_dim)

        self.perm = 1
        self.embed_dim = 200
        self.k_h = 20
        self.k_w = 10
        self.cross_patt = self.get_cross()

    # cross pattern
    def get_cross(self):
        ent_perm = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.perm)])
        rel_perm = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.perm)])
        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.k_h):
                for j in range(self.k_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
            comb_idx.append(temp)
        cross_patt = torch.LongTensor(np.int32(comb_idx))
        return cross_patt

    def head_reshape(self, x, b, i=1):
        x0 = x[0].reshape(-1, 1, 64, 243)
        x1 = x[1].reshape(-1, 1, 64, 243)
        x2 = x[2].reshape(-1, 1, 64, 243)
        x3 = x[3].reshape(-1, 1, 64, 243)
        # 64, 243 or 192 , 81
        return x0, x1, x2, x3

    def permute(self, x0, x1, x2, x3):
        x0 = x0.permute(1, 0, 2, 3)
        x1 = x1.permute(1, 0, 2, 3)
        x2 = x2.permute(1, 0, 2, 3)
        x3 = x3.permute(1, 0, 2, 3)
        return x0, x1, x2, x3

    def init(self):
        torch.nn.init.xavier_normal_(self.entity_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.relation_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.filter1.weight.data)
        torch.nn.init.xavier_normal_(self.filter3.weight.data)
        torch.nn.init.xavier_normal_(self.filter5.weight.data)

    def forward(self, entity_id, relation_id):
        # (b, 1, 200)
        entity = self.entity_embedding(entity_id).reshape(-1, 1, self.ent_dim)
        relation = self.relation_embedding(relation_id).reshape(-1, 1, self.rel_dim)

        sub_emb = self.entity_embedding(entity_id)
        rel_emb = self.relation_embedding(relation_id)

        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        cross_patt = comb_emb[:, self.cross_patt]

        f1 = self.filter1(relation_id)
        f1 = f1.reshape(entity.size(0) * self.in_channels * self.out_1, 1, 3, 3)
        f3 = self.filter3(relation_id)
        f3 = f3.reshape(entity.size(0) * self.in_channels * self.out_2, 1, 3, 3)
        f5 = self.filter5(relation_id)
        f5 = f5.reshape(entity.size(0) * self.in_channels * self.out_3, 1, 1, 9)

        # (b, 2, 200)→ (b, 200, 2)→ (b, 1, 20, 20)
        # S, A, C are commonly used to permute and reshape the embeddings from the translational perspective
        # x = torch.cat([entity, relation], 1).reshape(-1, 1, 20, 20) # S: stack pattern
        # x = torch.cat([entity, relation], 1).transpose(1, 2).reshape(-1, 1, 20, 20) # A: alternate pattern
        # x = cross_patt.reshape(-1, 1, 20, 20) # C: cross pattern
        x = torch.cat([entity, relation], 1).reshape(-1, 1, 20, 20)
        x = self.bn0(x)
        x = self.input_drop(x)

        # carry out the group equivariant convolution operation
        x = self.group_equivariant_conv1(x)
        x = torch.split(x, 1, dim=2)
        x00, x11, x22, x33 = self.head_reshape(x, entity.size(0))
        x00, x11, x22, x33 = self.permute(x00, x11, x22, x33)
        xx = x00 + x11 + x22 + x33

        # (1, b*in*out, H-kH+1, W-kW+1)
        # if embedding dimension is 400, set: groups=2 * entity.size(0)
        x1 = F.conv2d(xx, f1, groups=entity.size(0), padding=(3, 3), dilation=(3, 3))
        x1 = x1.reshape(entity.size(0), self.out_1, self.reshape_H, self.reshape_W)
        x1 = self.bn1_1(x1)

        # if embedding dimension is 400, set: groups=2 * entity.size(0)
        x3 = F.conv2d(xx, f3, groups=entity.size(0), padding=(1, 1))
        x3 = x3.reshape(entity.size(0), self.out_2, self.reshape_H, self.reshape_W)
        x3 = self.bn1_2(x3)

        # if embedding dimension is 400, set: groups=2 * entity.size(0)
        x5 = F.conv2d(xx, f5, groups=entity.size(0), padding=(0, 4))
        x5 = x5.reshape(entity.size(0), self.out_3, self.reshape_H, self.reshape_W)
        x5 = self.bn1_3(x5)

        x = torch.cat([x1, x3, x5], dim=1)
        x = torch.relu(x)
        x = self.feature_map_drop(x)

        # (b, fc_length)
        x = x.view(entity.size(0), -1)

        # (b, ent_dim)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.relu(x)

        # (batch, ent_dim)*(ent_dim, ent_num)=(batch, ent_num)
        x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


