import torch
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
from torch_geometric.utils import dense_to_sparse, degree
from utils import to_scipy
import math

device = 'cuda'
def compute_s_r(h, gamma, adj, num_relations, labels):
    output = h.to(device)
    means = output.mean(1, keepdim=True)
    deviations = output.std(1, keepdim=True)
    output = (output - means) / deviations
    output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)

    s_r = []

    for r in range(0, num_relations):
        edge_index = dense_to_sparse(adj[r])[0]
        row, col = edge_index
        norm_degree = degree(row, num_nodes=labels.shape[0]).clamp(min=1)
        norm_degree = torch.pow(norm_degree, -0.5)
        norm_degree_origin = torch.pow(norm_degree, -0.5)
        norm_degree_row = norm_degree_origin[row].unsqueeze(1).expand_as(output[row])
        norm_degree_col = norm_degree_origin[col].unsqueeze(1).expand_as(output[col])
        result_row = output[row] / norm_degree_row
        result_col = output[col] / norm_degree_col
        miu = torch.sum(torch.pow(torch.sum(torch.pow(result_row - result_col, 2), dim=1) + 1e-4,
                                  (2 - gamma)/2))
        s_r.append(miu/len(row))
    s_r = torch.reshape(torch.tensor(s_r), ((num_relations), 1)).to(device)
    return s_r


def initialization(num_relations, lp_list, h, num_nodes):
    with torch.no_grad():
        adj_list_sp = []
        laplace_list = []
        for r in range(num_relations):
            B = sparse_mx_to_torch_sparse_tensor(to_scipy(lp_list[r]))
            laplace_list.append(B)

    # Initialization coefficients and feature
    u = torch.full((num_relations, 1), 1 / (num_relations)).to(device)
    output = h.to(device)
    means = output.mean(1, keepdim=True)
    deviations = output.std(1, keepdim=True)
    output = (output - means) / deviations
    output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)

    # Calculating the initial value of totalvariation
    w = calculate_relational_totalvariation(num_relations, adj_list_sp, laplace_list, output, num_nodes)
    c_l1tr = torch.norm(w, p=1, dim=0).to(device)
    return w


def calculate_relational_totalvariation(num_relations, laplace_r, output, num_nodes):
    totalvariation = []

    for r in range(0, num_relations):
        a = []
        l = laplace_r[r].data.to(device)
        miu = calculate_totalvariation(a, l, output, num_nodes)
        totalvariation.append(miu)
        del a, l, miu
    w = torch.reshape(torch.tensor(totalvariation), ((num_relations), 1)).to(device)

    return w

def calculate_totalvariation(l_r, x):
    ft = torch.mm(l_r, x)
    f = torch.mm(ft.t(), x)
    miu = f.trace()
    if math.isnan(miu):
        miu = 0
    return miu
