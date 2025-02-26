import torch
import utils
import time
import numpy as np
import torch.nn.functional as F
import configparser
from utils import get_evaluation_results
from utils import load_data_Isogram, load_multi_view_data
from model import CGCN

def classification_multi(args):
    conf = configparser.ConfigParser()

    multirelational_dataset = ['ACM3025_0', 'DBLP_0', 'imdb5k_0', 'yelp_0']
    multi_att_mod_dataset = ['MNIST', 'HW', 'animals',
                         'BDGP', 'esp-game', 'flickr']

    all_ACC = []
    all_P = []
    all_R = []
    all_F1_ma = []
    all_F1_mi = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for i in range(args.n_repeated):
        if args.dataset in multirelational_dataset:
            config_path = './config_demo' + '.ini'
            conf.read(config_path)
            args.alpha = conf.getfloat(args.dataset, 'alpha')
            gamma = conf.get(args.dataset, 'gamma')
            numerator, denominator = map(int, gamma.split('/'))
            args.gamma = numerator / denominator
            args.lamda2 = conf.getint(args.dataset, 'lamda2')
            args.ratio = conf.getfloat(args.dataset, 'ratio')
            features, adj_list, labels, idx_train, idx_val, idx_test, num_relations, num_features, num_classes, num_nodes = load_data_Isogram(args, device)
        elif args.dataset in multi_att_mod_dataset:
            config_path = './config_demo' + '.ini'
            conf.read(config_path)
            args.alpha = conf.getfloat(args.dataset, 'alpha')
            gamma = conf.get(args.dataset, 'gamma')
            numerator, denominator = map(int, gamma.split('/'))
            args.gamma = numerator / denominator
            args.lamda2 = conf.getint(args.dataset, 'lamda2')
            args.ratio = conf.getfloat(args.dataset, 'ratio')
            features, adj_list, labels, idx_train, idx_val, idx_test, num_relations, num_features, num_classes, num_nodes = load_multi_view_data(args, device)
        else:
            print("We do not have {} dataset right now.".format(args.dataset))

        utils.set_seed(args.seed)
        #np.random.seed(args.seed)

        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)

        net = CGCN(args, adj_list, labels, num_features, num_nodes, num_relations, num_classes)
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        dur = []
        los = []
        counter = 0
        min_loss = 100.0
        max_acc = 0.0


        for epoch in range(args.max_epoch):
            if epoch >= 3:
                t0 = time.time()
            net.train()
            logp, _ = net(features)
            cla_loss = F.nll_loss(logp[idx_train], labels[idx_train])
            loss = cla_loss
            train_acc = utils.accuracy(logp[idx_train], labels[idx_train])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.eval()
            logp, _ = net(features)
            test_acc = utils.accuracy(logp[idx_test], labels[idx_test])
            labels_true = labels[idx_test]
            loss_val = F.nll_loss(logp[idx_val], labels[idx_val]).item()
            val_acc = utils.accuracy(logp[idx_val], labels[idx_val])
            pred_labels = torch.argmax(logp[idx_test], 1).cpu().detach().numpy()
            labels_true = labels_true.cpu().detach().numpy()
            test_R, test_P, test_F1_ma, test_F1_mi = get_evaluation_results(labels_true, pred_labels)
            los.append([epoch, loss_val, val_acc, test_acc, test_R, test_P, test_F1_ma, test_F1_mi])

            if loss_val < min_loss and max_acc < val_acc:
                min_loss = loss_val
                max_acc = val_acc
                counter = 0
            else:
                counter += 1

            if epoch >= 3:
                dur.append(time.time() - t0)

            print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
                epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))

        los.sort(key=lambda x: -x[2])
        acc = los[0][-5]
        all_ACC.append(acc)
        print("ACC: {:.2f}".format(acc*100))
        r = los[0][-4]
        all_R.append(r)
        print("R: {:.2f}".format(r*100))
        p = los[0][-3]
        all_P.append(p)
        print("P: {:.2f}".format(p*100))
        f1_ma = los[0][-2]
        all_F1_ma.append(f1_ma)
        print("F1_ma: {:.2f}".format(f1_ma*100))
        f1_mi = los[0][-1]
        all_F1_mi.append(f1_mi)
        print("F1_mi: {:.2f}".format(f1_mi * 100))
    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    print("F1_macro : {:.2f} ({:.2f})".format(np.mean(all_F1_ma) * 100, np.std(all_F1_ma) * 100))
    print("F1_micro : {:.2f} ({:.2f})".format(np.mean(all_F1_mi) * 100, np.std(all_F1_mi) * 100))


