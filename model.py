import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, dense_to_sparse
from coefficients import compute_s_r
import math

class CCLayer(MessagePassing):
    def __init__(self, args, adj_list, labels, num_nodes, num_relations):
        super(CCLayer, self).__init__(aggr='add')
        self.adj_list = adj_list
        self.dropout = nn.Dropout(args.dropout)
        self.gate = nn.Linear(2 * args.num_hidden, 1)
        self.gamma = args.gamma
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.labels = labels
        self.lamda1 = (1/args.alpha)-1
        self.lamda2 = args.lamda2
        self.threshold = args.threshold
        self.threshold_c = args.threshold_c
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def mp(self, adj, h):
        edge_index = dense_to_sparse(adj)[0]
        row, col = edge_index
        norm_degree = degree(row, num_nodes=self.labels.shape[0]).clamp(min=1)
        norm_degree = torch.pow(norm_degree, -0.5)
        norm_degree_origin = torch.pow(norm_degree, -0.5)
        norm_degree_row = norm_degree_origin[row].unsqueeze(1).expand_as(h[row])
        norm_degree_col = norm_degree_origin[col].unsqueeze(1).expand_as(h[col])
        result_row = h[row] / norm_degree_row
        result_col = h[col] / norm_degree_col
        g = torch.tanh(1 / torch.pow(torch.sum(torch.pow(result_row - result_col, 2), dim=1) + 1e-4,
                                     self.gamma/2))
        norm = g * norm_degree[row] * norm_degree[col]
        norm = self.dropout(norm)
        out = self.propagate(edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)
        return out

    def comp_coefficient(self, h):
        device = 'cuda'
        # Initialization
        s_r = compute_s_r(h, self.gamma, self.adj_list, self.num_relations, self.labels)
        u = torch.full((self.num_relations, 1), 1 / (self.num_relations)).to(device)
        l1tr = torch.norm(s_r, p=1, dim=0).to(device)

        fi = l1tr + ((2 * self.lamda2) / self.lamda1)
        condition = 0
        t = 1
        i =0
        while condition >= 0:
            i +=1
            # u_before = copy.deepcopy(u)
            u_before = u.detach().clone()
            T_t = ((2 * math.log(self.num_relations)) / (t * (fi * fi))).sqrt()
            f_de = (((2 * self.lamda2) / self.lamda1) * u.to(device)) + s_r.to(device)
            u_ta = torch.mul(u.to(device), torch.exp(-T_t * f_de).to(device))
            t = t + 1
            for r in range(self.num_relations):
                u_tamp = u_ta[r]
                u_tampm = torch.sum(u_ta)
                u_tnext = u_tamp / u_tampm
                u[r] = u_tnext
            condition = torch.sqrt(torch.sum(torch.square(u_before - u)))
            if condition.item() < self.threshold_c:
                break
        return u

    def forward(self, h):
        #Update embedding
        u = self.comp_coefficient(h)
        f_out = torch.zeros_like(h)
        for r in range(self.num_relations):
            out = self.mp(self.adj_list[r], h)
            f_out = f_out + out * u[r]
        return f_out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out




class CGCN(nn.Module):
    def __init__(self, args, adj_list, labels, num_features, num_nodes, num_relations, num_classes):
        super(CGCN, self).__init__()
        self.alpha = args.alpha
        self.layer_num = args.layer_num
        self.dropout = args.dropout
        self.layers = nn.ModuleList()
        self.adj_list = adj_list
        for i in range(self.layer_num):
            self.layers.append(CCLayer(args, adj_list, labels, num_nodes, num_relations))
        self.t1 = nn.Linear(num_features, args.num_hidden)
        self.t2 = nn.Linear(args.num_hidden, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.alpha * raw + (1-self.alpha) * h
        h = self.t2(h)
        return F.log_softmax(h, 1), h