import sys
sys.path.append('../')
import dgl
import dgl.function as fn
import os
import multiprocessing as mp
from tqdm import tqdm
import pdb
import random
import numpy as np
import torch
import torch.nn as nn
import logging
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
from utils.parser import parse_args
from dgl.nn.pytorch.conv import GraphConv
from utils.dataloader_steam import Dataloader_steam_filtered

from utils.dataloader_item import Dataloader_item_graph
from models.model import Proposed_model
from models.model import SSLoss
from models.Predictor import Predictor
import pickle
import torch.nn.functional as F
import time


ls_5 = []
ls_10 = []
ls_20 = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_valid_mask(DataLoader, graph, valid_user):
    data_exist_dir = os.path.join("data_exist")
    path_valid_mask_trail = path+"/valid_mask.pth"
    if os.path.exists(path_valid_mask_trail):
        valid_mask = torch.load(path_valid_mask_trail)
        return valid_mask
    else:
        valid_mask = torch.zeros(len(valid_user), graph.num_nodes('game'))
        for i in range(len(valid_user)):
            user = valid_user[i]
            item_train = torch.tensor(DataLoader.dic_user_game[user])
            valid_mask[i, :][item_train] = 1
        valid_mask = valid_mask.bool()
        torch.save(valid_mask, path_valid_mask_trail)
        return valid_mask

def construct_negative_graph(graph, etype,device):

    utype, _ , vtype = etype
    src, _ = graph.edges(etype = etype)
    src = src.to(device)
    dst = torch.randint(graph.num_nodes(vtype), size = src.shape).to(device)
    return dst, dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

def get_coverage(ls_tensor, mapping):
    covered_items = set()
    
    for i in ls_tensor:
        if int(i) in mapping.keys():
            types = mapping[int(i)]
            covered_items = covered_items.union(set(types))
    
    return float(len(covered_items))

def validate(valid_mask, dic, h, ls_k, mapping, to_get_coverage,device):
    users = torch.tensor(list(dic.keys())).long()
    user_embedding = h['user'][users]
    game_embedding = h['game']
    rating = torch.mm(user_embedding, game_embedding.t())
    rating[valid_mask] = -float('inf')
    valid_mask = torch.zeros_like(valid_mask)
    for i in range(users.shape[0]):
        user = int(users[i])
        items = torch.tensor(dic[user])
        valid_mask[i, items] = 1
    _, indices = torch.sort(rating, descending = True)

    indices = indices.to(device)
    valid_mask = valid_mask.to(device)
    ls = [valid_mask[i,:][indices[i, :]] for i in range(valid_mask.shape[0])]
    result = torch.stack(ls).float()
    res = []
    ndcg = 0
    for k in ls_k:
        discount = (torch.tensor([i for i in range(k)]) + 2).log2()
        ideal, _ = result.sort(descending = True)
        ideal = ideal.to(device)
        discount = discount.to(device)
        idcg = (ideal[:, :k] / discount).sum(dim = 1)
        dcg = (result[:, :k] / discount).sum(dim = 1)
        ndcg = torch.mean(dcg / idcg)
        
        recall = torch.mean(result[:, :k].sum(1) / result.sum(1))
        hit = torch.mean((result[:, :k].sum(1) > 0).float())
        precision = torch.mean(result[:, :k].mean(1))
        
        if to_get_coverage == False:
            coverage = -1
        else:
            cover_tensor = torch.tensor([get_coverage(indices[i,:k], mapping) for i in range(users.shape[0])])
            coverage = torch.mean(cover_tensor)
        
        logging_result = "For k = {}, ndcg = {}, recall = {}, hit = {}, precision = {}, coverage = {}".format(
            k, ndcg, recall, hit, precision, coverage)
        logging.info(logging_result)
        res.append(logging_result)
    
    return coverage, str(res)

if __name__ == '__main__':

    seed=int(2025)
    setup_seed(seed)
    args = parse_args()
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    path = "./steam_data"

    user_id_path = path + '/users.txt'
    app_id_path = path + '/app_id.txt'
    genres_path = path + '/Games_Genres.txt'

    DataLoader = Dataloader_steam_filtered(args, path, user_id_path, app_id_path, genres_path)
    graph = DataLoader.graph.to(device)
    DataLoader_item = Dataloader_item_graph( app_id_path, genres_path, DataLoader)
    graph_item_item = DataLoader_item.graph_item

    graph = dgl.edge_type_subgraph(graph, [('user','play','game'),('game','played by','user')])
    

    
    valid_user = list(DataLoader.valid_data.keys())
    valid_mask = get_valid_mask(DataLoader, graph, valid_user)

    model = Proposed_model(args, graph, graph_item_item,
                        device, gamma=args.gamma, ablation = False)
    model.to(device)    
    ssloss = SSLoss(args)
    predictor = Predictor()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    stop_count = 0
    ls_k = args.k
    total_epoch = 0  
    loss_pre = float('inf')
    loss = 0
    test_result = None
    coverage = 0

    batch_size=args.ssl_batch_size
    n_users=60742
    n_batch = (n_users + batch_size - 1) // batch_size
    
    start_time = time.time()
    print_interval = 6  



    for epoch in range(args.epoch):
        model.train()
        dst, graph_neg = construct_negative_graph(graph,('user','play','game'),device)
        h,h_sub1,h_sub2 = model()
        ssloss_value=0

        for idx in range(n_batch):

            ua_embeddings_sub1,ia_embeddings_sub1=h_sub1['user'],h_sub1['game']
            ua_embeddings_sub2,ia_embeddings_sub2=h_sub2['user'],h_sub2['game']            
            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, n_users)

            batch_ua_embeddings_sub1 = ua_embeddings_sub1[start_idx:end_idx]
            batch_ia_embeddings_sub1 = ia_embeddings_sub1[start_idx:end_idx]
            batch_ua_embeddings_sub2 = ua_embeddings_sub2[start_idx:end_idx]
            batch_ia_embeddings_sub2 = ia_embeddings_sub2[start_idx:end_idx]


            ssloss_value =ssloss_value + ssloss.forward(batch_ua_embeddings_sub1, batch_ua_embeddings_sub2, batch_ia_embeddings_sub1, batch_ia_embeddings_sub2,device)

  
        score = predictor(graph, h, ('user','play','game'))
        score_neg = predictor(graph_neg, h, ('user','play','game'))
        loss_pre = loss
        eps=1e-15
        loss = -torch.sum(torch.log(torch.sigmoid(score - score_neg)+eps))
        #score_neg_reweight = score_neg * (1 / (1 + torch.exp(-score_neg*args.balance)) * args.K)
        #loss =  (-((score - score_neg_reweight).sigmoid().clamp(min=1e-8, max=1-1e-8).log())).sum()

        loss = loss.to(device)
        total_loss=loss+ssloss_value*args.ssl_loss_weight
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        total_epoch += 1
        
        to_get_coverage = True

        if total_epoch > 0:
            print("="*40)
            logging.info('Epoch {}'.format(epoch))
            logging.info(f"loss = {loss}\n")
            model.eval()
            logging.info("begin validation")

            _, result = validate(valid_mask, DataLoader.valid_data, h, ls_k, DataLoader_item.genre,to_get_coverage,device)
            logging.info(result)

            if loss < loss_pre:
                stop_count = 0
                logging.info("begin test")
                _, test_result = validate(valid_mask, DataLoader.test_data, h, ls_k, DataLoader_item.genre, to_get_coverage,device)
                logging.info(test_result)
            else:
                stop_count += 1
                logging.info(f"stop count:{stop_count}")
                if stop_count > args.early_stop:
                    logging.info('early stop')
                    break
 
    logging.info(test_result)

    torch.save(model, path_model)
