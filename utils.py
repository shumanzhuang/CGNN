import torch
import numpy as np
import torch_geometric
import random
import os
from scipy import io
import scipy.sparse as ss
from sklearn.model_selection import train_test_split
import scipy.io as sio
import argparse
from sklearn import metrics
import time
from sklearn.neighbors import NearestNeighbors

def get_evaluation_results(labels_true, labels_pred):
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    F1_macro = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_micro = metrics.f1_score(labels_true, labels_pred, average='micro')

    return R, P, F1_macro, F1_micro


def loadMatData(dataset_name, feature_normalize=False):
    '''
    return features: Tensor
    edges: Sparse Tensor
    edge_weights: Sparse Tensor
    gnd: Tensor
    '''
    DATAPATH = './datasets/single_view'
    data = io.loadmat(f'{DATAPATH}/{dataset_name}.mat')
    features = data['X']  # .dtype = 'float32'
    features = torch.from_numpy(features).float()
    # features = normalize(features)

    gnd = data['Y']
    label = np.array(data['Y'], dtype=np.int32).flatten()
    num_features = features.shape[1]
    num_classes = np.max(label) + 1

    #keys = data.keys()
    #print(keys)

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    idx_train = torch.tensor(data['train']).to(torch.int64)
    idx_val = torch.tensor(data['val']).to(torch.int64)
    idx_test = torch.tensor(data['test']).to(torch.int64)


    gnd = gnd.flatten()
    if np.min(gnd) == 1:
        gnd = gnd - 1
    gnd = torch.from_numpy(gnd)

    adj = data['adj_t']
    adj = torch.from_numpy(adj)
    adj = adj + adj.t().multiply(adj.t() > adj) - adj.multiply(adj.t() > adj)
    edge_index = np.argwhere(adj > 0)
    adj = normalize_adj(adj)

    if feature_normalize == 1:
        print("Feature Normalized.")
        features = normalize(features)
    x = features
    y = torch.tensor(label, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
    return data, features, adj, edge_index, gnd, num_features, num_classes, idx_train, idx_val, idx_test

def load_data(args, device):
    data = sio.loadmat(args.path + args.dataset + '.mat')
    print("data: ", data)
    if args.dataset == 'DBLP4057':
        feature = data['features']
    else:
        feature = data['feature']
    adj_hat_list = []
    lp_hat_list = []

    labels = np.argmax(data['label'], axis=1).flatten()
    labels = labels - min(set(labels))
    labels = torch.from_numpy(labels).long()
    num_classes = len(np.unique(labels))

    train, val, test = len(data['train_idx'].tolist()[0]), len(data['val_idx'].tolist()[0]), len(data['test_idx'].tolist()[0])
    idx_train, idx_val, idx_test = generate_partition_heter(labels, train, val, test, args.shuffle_seed)

    feature = torch.from_numpy(feature).float().to(device)

    if args.dataset == 'DBLP4057':
        for key in ['net_APTPA', 'net_APCPA', 'net_APA']:
            adj = data[key]
            adj = ss.coo_matrix(adj)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj_hat, lp_hat = construct_adj_hat(adj)
            adj_hat = torch.from_numpy(adj_hat.todense()).float().to(device)
            lp_hat = torch.from_numpy(lp_hat.todense()).float().to(device)
            adj_hat_list.append(adj_hat)
            lp_hat_list.append(lp_hat)
    elif args.dataset == 'imdb5k':
        for key in ['MAM', 'MDM', 'MYM']:
            adj = data[key]
            adj = ss.coo_matrix(adj)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj_hat, lp_hat = construct_adj_hat(adj)
            adj_hat = torch.from_numpy(adj_hat.todense()).float().to(device)
            lp_hat = torch.from_numpy(lp_hat.todense()).float().to(device)
            adj_hat_list.append(adj_hat)
            lp_hat_list.append(lp_hat)
    else:
        for key in ['PAP', 'PLP']:
            adj = data[key]
            adj = ss.coo_matrix(adj)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj_hat, lp_hat = construct_adj_hat(adj)
            adj_hat = torch.from_numpy(adj_hat.todense()).float().to(device)
            lp_hat = torch.from_numpy(lp_hat.todense()).float().to(device)
            adj_hat_list.append(adj_hat)
            lp_hat_list.append(lp_hat)


    idx_train = torch.tensor(idx_train)
    idx_val = torch.tensor(idx_val)
    idx_test = torch.tensor(idx_test)
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    num_relations = len(adj_hat_list)
    num_features = feature.shape[1]
    num_nodes = feature.shape[0]
    return feature, adj_hat_list, lp_hat_list, labels, idx_train, idx_val, idx_test, num_relations, num_features, num_classes, num_nodes


def load_data_Isogram(args, device):
    args.path = './datasets/multi_relational/'
    data = sio.loadmat(args.path + args.dataset + '.mat')
    features = data['X']
    feature = torch.from_numpy(features).float().to(device)
    adj_raw = data['adj']
    adj_list = []
    #adj_t_list = []
    labels = np.array(data['Y'].flatten())
    labels = labels - min(set(labels))

    idx_labeled_train, idx_unlabeled, idx_labeled_val = generate_partition(labels, args.ratio, args.shuffle_seed)
    labels = torch.tensor(labels)
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    for i in range(adj_raw.shape[1]):
        if (args.dataset == 'ACM3025_0' and i == 2):
            break
        try:
            adj = adj_raw[0][i].todense()
        except:
            adj = adj_raw[0][i]
        adj = ss.coo_matrix(adj)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj_hat, _ = construct_adj_hat(adj)
        adj_hat = torch.from_numpy(adj_hat.todense()).float().to(device)
        adj_list.append(adj_hat)

    idx_train = torch.tensor(idx_labeled_train)
    idx_val = torch.tensor(idx_labeled_val)
    idx_test = torch.tensor(idx_unlabeled)
    num_features = feature.shape[1]
    num_nodes = feature.shape[0]
    num_relations = len(adj_list)
    return feature, adj_list, labels, idx_train, idx_val, idx_test, num_relations, num_features, num_classes, num_nodes

def load_multi_view_data(args, device):
    args.path = './datasets/multi_attribute_modality/'
    data = sio.loadmat(args.path + args.dataset + '.mat')
    features = data['X']
    feature_list = []
    adj_list = []

    if args.dataset == 'HW':
        labels = data['truth'].flatten()
    else:
        labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    idx_labeled_train, idx_unlabeled, idx_labeled_val = generate_partition(labels, args.ratio, args.shuffle_seed)
    labels = torch.from_numpy(labels).long()
    num_classes = len(np.unique(labels))

    for i in range(features.shape[1]):
        # print("Loading the data of" + str(i) + "th view")
        #features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix_csr(feature):
            feature = feature.todense()
            print("sparse")
        direction_judge = './adj_matrix/' + args.dataset + '/' + 'v' + str(i) + '_knn' + str(args.knns) + '_adj.npz'
        if os.path.exists(direction_judge):
            print("Loading the adjacency matrix of " + str(i) + "th view of " + args.dataset)
            adj = torch.from_numpy(ss.load_npz(direction_judge).todense()).float().to(device)
        else:
            print("Constructing the adjacency matrix of " + str(i) + "th view of " + args.dataset)
            adj = construct_adjacency_matrix(feature, args.knns, args.pr1, args.pr2, args.common_neighbors)
            adj = ss.coo_matrix(adj)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            #lp = construct_laplacian(adj)
            save_direction = './adj_matrix/' + args.dataset + '/'
            if not os.path.exists(save_direction):
                os.makedirs(save_direction)
            print("Saving the adjacency matrix to " + save_direction)
            ss.save_npz(save_direction + 'v' + str(i) + '_knn' + str(args.knns) + '_adj.npz', adj)
            #lp = torch.from_numpy(lp.todense()).float().to(device)
        adj = to_scipy(adj)
        adj_hat, _ = construct_adj_hat(adj)
        adj_hat = torch.from_numpy(adj_hat.todense()).float().to(device)
        adj_list.append(adj_hat)

        if args.dataset in ['GRAZ02', 'Out_Scene']:
            feature = torch.from_numpy(feature.astype(np.float32)).to(device)
        else:
            feature = torch.from_numpy(feature).float().to(device)
        feature_list.append(feature)

    feature =  torch.cat(feature_list, 1)
    labels = labels.to(device)
    num_features= feature.shape[1]
    num_nodes= feature.shape[0]
    num_relations = len(adj_list)
    idx_train = torch.tensor(idx_labeled_train)
    idx_val = torch.tensor(idx_labeled_val)
    idx_test = torch.tensor(idx_unlabeled)
    return feature, adj_list, labels, idx_train, idx_val, idx_test, num_relations, num_features, num_classes, num_nodes

def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(features)
    adj_construct = nbrs.kneighbors_graph(features)  # <class 'scipy.sparse.csr.csr_matrix'>
    adj = ss.coo_matrix(adj_construct)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if prunning_one:
        # Pruning strategy 1
        original_adj = adj.A
        judges_matrix = original_adj == original_adj.T
        adj = original_adj * judges_matrix
        adj = ss.csc_matrix(adj)
    # obtain the adjacency matrix without self-connection
    adj = adj - ss.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = ss.coo_matrix(adj)
        adj.eliminate_zeros()

    print("The construction of Laplacian matrix is finished!")
    print("The time cost of construction: ", time.time() - start_time)
    adj = ss.coo_matrix(adj)
    return adj

def construct_laplacian(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = ss.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_ = ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1)) # <class 'numpy.ndarray'> (n_samples, 1)
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    lp = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocsr()
    return lp

def generate_partition(labels, ratio, seed):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num_train = {}  # number of labeled samples for each class
    total_num = int (round(ratio * len(labels)))
    val_num = int (round( 0.1 * len(labels)))
    for label in each_class_num.keys():
        labeled_each_class_num_train[label] = max(int (ratio * each_class_num[label]), 1)  # min is 1
    labeled_each_class_num_val = {}  # number of labeled samples for each class
    for label in each_class_num.keys():
        labeled_each_class_num_val[label] = max(int (0.1 * each_class_num[label]), 1)  # min is 1
    print(labeled_each_class_num_train)
    print(labeled_each_class_num_val)
    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    p_val = []
    index = [i for i in range(len(labels))]
    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num_train[label] > 0):
            labeled_each_class_num_train[label] -= 1
            p_labeled.append(index[idx])
            total_num -= 1
        else:
            if (labeled_each_class_num_val[label] > 0):
                labeled_each_class_num_val[label] -= 1
                p_val.append(index[idx])
                val_num -= 1
            else:
                p_unlabeled.append(index[idx])

    return p_labeled, p_unlabeled, p_val

def construct_adj_hat(adj):
    """
        construct the Laplacian matrix
    :param adj: original adj matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    # adj = ss.coo_matrix(adj)
    adj_ = ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1)) # <class 'numpy.ndarray'> (n_samples, 1)
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    lp = ss.eye(adj.shape[0]) - adj_wave
    return adj_wave, lp

def generate_partition_heter(labels, train, val, test, sf_seed):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    # for label in each_class_num.keys():
    #     # labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1) # min is 1
    #     labeled_each_class_num[label] = num_perclass
    # index of labeled and unlabeled samples
    num_train = train
    num_test = test
    num_val = val
    idx_train = []
    idx_test = []
    idx_val = []
    index = [i for i in range(len(labels))]
    # print(index)
    if sf_seed > 0:
        random.seed(sf_seed)
        random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if num_train > 0:
            num_train -= 1
            idx_train.append(index[idx])
        elif num_test > 0:
            num_test -= 1
            idx_test.append(index[idx])
        elif num_val > 0:
            num_val -= 1
            idx_val.append(index[idx])
    print('train: {}, val: {}, test: {}'.format(len(idx_train), len(idx_val), len(idx_test)))
    return idx_train, idx_val, idx_test

def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    return seed


def split_nodes(labels, train_ratio, val_ratio, test_ratio, random_state, split_by_label_flag):
    idx = torch.arange(labels.shape[0])
    if split_by_label_flag:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio, stratify=labels)
    else:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio)

    if val_ratio:
        labels_train_val = labels[idx_train]
        if split_by_label_flag:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio), stratify=labels_train_val)
        else:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio))
    else:
        idx_val = None

    return idx_train, idx_val, idx_test


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item()*1.0/len(labels)

#def get_evaluation_results(labels_pred, labels_true):
#    R, P, F1 = groundtruth_metrics(labels_pred, labels_true, metrics=["recall", "precision", "f1_score"])
#    return R, P, F1

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool).cuda()
    mask[index] = 1
    return mask

def mask_to_index(mask):
    index = torch.where(mask == True)[0].cuda()
    return index

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = ss.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    # if isinstance(mx, np.ndarray):
    #     return torch.from_numpy(mx)
    # else:
    return mx

def normalize_adj(adj):
    # """Row-normalize sparse matrix"""
    # rowsum = np.array(mx.sum(1))
    # r_inv = np.power(rowsum, -0.5).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # mx = r_mat_inv.dot(mx)
    # if isinstance(mx, np.ndarray):
    #     return torch.from_numpy(mx)
    # else:
    #     return mx


    # D = torch.diag(adj.sum(1))
    # D_inv = torch.pow(D + torch.eye(adj.shape[0], adj.shape[0]), -1)
    # D_inv[torch.isinf(D_inv)] = 0.
    # A_hat = D_inv.matmul(adj + torch.eye(adj.shape[0], adj.shape[0]))
    # return A_hat + torch.diag(torch.diag(A_hat))

    """Row-normalize sparse matrix"""
    mx = np.array(adj + torch.eye(adj.shape[0], adj.shape[1]))
    rowsum = np.array(adj.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum + 1, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = ss.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx + np.diag(np.diag(mx))
    if isinstance(mx, np.ndarray):
        return torch.from_numpy(mx)
    else:
        return mx

def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return ss.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return ss.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.
    Args:
        tensor : torch.Tensor
                 given tensor
    Returns:
        bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False