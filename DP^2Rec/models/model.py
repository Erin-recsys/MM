import torch.nn as nn
from tqdm import tqdm
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv
import torch.nn.functional as F

import os
class Proposed_model(nn.Module):
    def __init__(self, args, graph, graph_item, device, gamma=80, ablation=False):
        super().__init__()
        print("\n=== Graph Information ===")
        print("Node types:", graph.ntypes)  
        print("Edge types:", graph.etypes)  
        print("Canonical edge types:", graph.canonical_etypes)  
        print("\n=== Node Statistics ===")
        for ntype in graph.ntypes:
            print(f"Number of {ntype} nodes:", graph.number_of_nodes(ntype))
        print("\n=== Edge Statistics ===")
        for etype in graph.etypes:
            print(f"Number of {etype} edges:", graph.number_of_edges(etype))
        self.ablation = ablation
        self.device_ = torch.device(device)
        torch.cuda.empty_cache()
        self.args = args
        self.param_decay = args.param_decay
        self.hid_dim = args.embed_size  
        self.attention_and = args.attention_and
        self.layer_num_and = args.layers_and
        self.layer_num_oridn=3  
        self.layer_num_or = args.layers_or   
        self.layer_num_user_game = args.layers_user_game
        self.graph_item = graph_item.to(self.device_)

        self.graph = graph.to(self.device_)
        self.ori_item2user = dgl.edge_type_subgraph(self.graph,['played by']).to(self.device_)
        self.ori_user2item = dgl.edge_type_subgraph(self.graph,['play']).to(self.device_)
        self.graph_item2user = dgl.edge_type_subgraph(self.graph,['played by']).to(self.device_)
        self.graph_user2item = dgl.edge_type_subgraph(self.graph,['play']).to(self.device_)








        self.edge_node_weight =True
        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim)).to(torch.float32)
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('game').shape[0], self.hid_dim)).to(torch.float32)



        
        seek_graph_path = f"/mnt/data/zhangjingmao/MM/seek-PDGRec-main/graph_and_weights/woSim1605diversity_exploration_max_per_genre{args.max_per_genre}.bin"
        seek_plays_weight_path = f"/mnt/data/zhangjingmao/MM/seek-PDGRec-main/graph_and_weights/woSim1605normalized_seek_plays_weights_max_per_genre{args.max_per_genre}.pth"
        print(f"Loading seek graph from: {seek_graph_path}")
        print(f"Loading seek weights from: {seek_plays_weight_path}")




        self.conv1 = GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True).to(self.device_)

        self.build_model_item(self.graph_item)
        self.build_model_ssl()
        self.build_model_dn()


        dn_graph_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "data_exist/dn_graph_em.bin")

        dn_graph, _ = dgl.load_graphs(dn_graph_path)

        seek_graph, _ = dgl.load_graphs(seek_graph_path)
        self.graph_dn = dn_graph[0].to(self.device_)

        self.graph_seek=seek_graph[0].to(self.device_)
        self.graph_dn = dgl.edge_type_subgraph(self.graph_dn, [('user','play','game'),('game','played by','user')])

        self.graph_seek = dgl.edge_type_subgraph(self.graph_seek, [('user','plays','game'),('game','played_by','user')])
        print("Successfully loaded dn graph from file")

        print("Successfully loaded seek graph from file")

        self.graph_item2user_dn = dgl.edge_type_subgraph(self.graph_dn,['played by']).to(self.device_)
        self.graph_user2item_dn = dgl.edge_type_subgraph(self.graph_dn,['play']).to(self.device_)

        self.graph_item2user_seek = dgl.edge_type_subgraph(self.graph_seek,['played_by']).to(self.device_)
        self.graph_user2item_seek = dgl.edge_type_subgraph(self.graph_seek,['plays']).to(self.device_)


        self.weight_edge = torch.load(seek_plays_weight_path).to(self.device_)



    def build_model_item(self, graph_item):
        self.sub_g1 = dgl.edge_type_subgraph(graph_item,['co_genre']).to(self.device_)


    def build_model_ssl(self):
        self.layers = nn.ModuleList()
        for _ in range(self.layer_num_user_game):
            layer = 0
            if self.edge_node_weight == True:
                layer = GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
            else:
                layer = dgl.nn.HeteroGraphConv({
                    'play': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True),
                    'played by': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
                })
            self.layers.append(layer)
        self.layers.to(self.device_)
    def build_model_dn(self):
        self.layers_dn = nn.ModuleList()
        for _ in range(self.layer_num_user_game):
            layer_dn = 0
            if self.edge_node_weight == True:
                layer_dn = GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
            else:
                layer_dn = dgl.nn.HeteroGraphConv({
                    'play': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True),
                    'played by': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
                })
            self.layers_dn.append(layer_dn)
        self.layers_dn.to(self.device_)


   
           
    def forward(self):
        h = {'user':self.user_embedding.clone(), 'game':self.item_embedding.clone()}
        for layer in self.layers:
            if self.edge_node_weight == True:
                h_user = layer(self.graph_item2user, (h['game'],h['user']))
                h_item = layer(self.graph_user2item, (h['user'],h['game']))
                h_user_seek=layer(self.graph_item2user_seek, (h['game'],h['user']),edge_weight=self.weight_edge)
                h_item_seek=layer(self.graph_user2item_seek, (h['user'],h['game']))
                h['user'] = h_user*self.args.weight_self+h_user_seek*self.args.weight_seek
                h['game'] = h_item*self.args.weight_self+h_item_seek*self.args.weight_seek
            else:
                h = layer(self.graph,h)
        h_sub1 = h
        h1= {'user':self.user_embedding.clone(), 'game':self.item_embedding.clone()}
        for layer_dn in self.layers_dn:
            if self.edge_node_weight == True:
                h_user = layer_dn(self.graph_item2user_dn, (h1['game'],h1['user']))
                h_item = layer_dn(self.graph_user2item_dn, (h1['user'],h1['game']))
                h_user_seek=layer(self.graph_item2user_seek, (h['game'],h['user']),edge_weight=self.weight_edge)
                h_item_seek=layer(self.graph_user2item_seek, (h['user'],h['game']))
                h1['user'] = h_user*self.args.weight_self+h_user_seek*self.args.weight_seek
                h1['game'] = h_item*self.args.weight_self+h_item_seek*self.args.weight_seek
            else:
                h1 = layer(self.graph,h)


        return h,h_sub1,h1
    


class SSLoss():
    def __init__(self,args):
        super(SSLoss, self).__init__()
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = 1
        self.ssl_game_weight=args.ssl_game_weight


    def forward(self,ua_embeddings_sub1, ua_embeddings_sub2, ia_embeddings_sub1,
                ia_embeddings_sub2,device):
        user_emb1 = ua_embeddings_sub1
        user_emb2 = ua_embeddings_sub2  
        normalize_user_emb1 = F.normalize(user_emb1, dim=1)
        normalize_user_emb2 = F.normalize(user_emb2, dim=1)
        normalize_all_user_emb2 = F.normalize(ua_embeddings_sub2, dim=1)
        pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2),
                                    dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)

        ttl_score_user = torch.matmul(normalize_user_emb1,
                                        normalize_all_user_emb2.T)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)  

        ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        item_emb1 = ia_embeddings_sub1
        item_emb2 = ia_embeddings_sub2

        normalize_item_emb1 = F.normalize(item_emb1, dim=1)
        normalize_item_emb2 = F.normalize(item_emb2, dim=1)
        normalize_all_item_emb2 = F.normalize(ia_embeddings_sub2, dim=1)
        pos_score_item = torch.sum(torch.mul(normalize_item_emb1, normalize_item_emb2), dim=1)
        ttl_score_item = torch.matmul(normalize_item_emb1, normalize_all_item_emb2.T)

        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)

        ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))*self.ssl_game_weight

        loss=ssl_loss_item+ssl_loss_user
        return loss
