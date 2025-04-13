import os
import sys
from dgl.data.utils import save_graphs
from tqdm import tqdm
from scipy import stats
import pdb
import torch
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import numpy as np
import dgl
from dgl.data import DGLDataset
import pandas as pd
from sklearn import preprocessing
import pickle

import dgl.function as fn
import pandas as pd

game_num = 7726

class Dataloader_steam_filtered(DGLDataset):
    def __init__(self, args, path, user_id_path, app_id_path,  genre_path, device = 'cpu', name = 'steam'):
        
        logging.info("steam dataloader init...")

        self.args = args
        self.path = path
        self.user_id_path = self.path+"/users.txt"


        self.app_id_path = self.path+"/app_id.txt"

        self.genre_path = self.path+"/Games_Genres.txt"


        self.train_game_path = self.path+"/train_game.txt"
        self.valid_game_path = self.path+"/valid_data/valid_game.txt"
        self.test_game_path = self.path+"/test_data/test_game.txt"
        self.train_time_path = self.path+"/train_time.txt"
        self.device=device
        self.graph_path = self.path + "/graph.bin"
        
        '''get user id mapping and app id mapping'''
        logging.info("reading user id mapping:")
        self.user_id_mapping = self.read_user_id_mapping(self.user_id_path)
        logging.info("reading app id mapping:")
        self.app_id_mapping = self.read_app_id_mapping(self.app_id_path)
        
        


        '''build valid and test data'''

        logging.info("build valid data:")
        self.valid_data = self.build_valid_data(self.valid_game_path)
        logging.info("build test data:")
        self.test_data = self.build_test_data(self.test_game_path)

        self.process()
        dgl.save_graphs(self.graph_path, self.graph)

    def generate_percentile(self, ls):
        dic = {}
        for ls_i in ls:  
            if ls_i[1] in dic:
                dic[ls_i[1]].append(ls_i[2])  
            else:
                dic[ls_i[1]] = [ls_i[2]]
        

        for key in tqdm(dic):
            dic[key] = sorted([time for time in dic[key] if time is not None and time != -1])
        

        dic_percentile = {}
        for key in tqdm(dic):
            dic_percentile[key] = {}
            length = len(dic[key])  
            for i in range(length):
                time = dic[key][i]
                dic_percentile[key][time] = (i + 1) / length  
        user_percentiles = {}
        for ls_i in ls:
            user, game, time = ls_i[0], ls_i[1], ls_i[2]
            if time is not None and time != -1:  
                if user not in user_percentiles:
                    user_percentiles[user] = []
                user_percentiles[user].append(dic_percentile[game][time])
        
        
        user_mean_percentile = {
            user: np.mean(percentiles) if percentiles else None
            for user, percentiles in user_percentiles.items()
        }
        
        
        for i in tqdm(range(len(ls))):
            user, game, time = ls[i][0], ls[i][1], ls[i][2]
            if time is not None and time != -1:
                ls[i].append(dic_percentile[game][time])  
            else:
                
                ls[i].append(user_mean_percentile[user] if user_mean_percentile[user] is not None else 0)
        
        return ls  

    def read_user_id_mapping(self, path):
        mapping = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_user_id_mapping = os.path.join(base_dir, "data_exist/user_id_mapping.pkl")
        if os.path.exists(path_user_id_mapping):
            with open(path_user_id_mapping, 'rb') as f:
                mapping = pickle.load(f)

        else:
            count = int(0)
            with open(path,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line not in mapping.keys():
                        mapping[line] = int(count)
                        count += 1
            with open(path_user_id_mapping, 'wb') as f:
                pickle.dump(mapping, f)
        return mapping



    def read_app_id_mapping(self, path):
        mapping = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_app_id_mapping = os.path.join(base_dir, "data_exist/app_id_mapping.pkl")
        if os.path.exists(path_app_id_mapping):
            with open(path_app_id_mapping, 'rb') as f:
                mapping = pickle.load(f)

        else:
            count = int(0)
            with open(path,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line not in mapping.keys():
                        mapping[line] = int(count)
                        count += 1
            with open(path_app_id_mapping, 'wb') as f:
                pickle.dump(mapping, f)
        return mapping


    def build_valid_data(self, path):
        intr = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
        path_valid_data = os.path.join(base_dir, "data_exist/valid_data.pkl")
        if os.path.exists(path_valid_data):
            with open(path_valid_data, 'rb') as f:
                intr = pickle.load(f)
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')
                    user = self.user_id_mapping[line[0]]

                    if user not in intr:
                        intr[user] = [self.app_id_mapping[game] for game in line[1:]]
            with open(path_valid_data, 'wb') as f:
                pickle.dump(intr, f)
        return intr



    def build_test_data(self, path):
        intr = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_valid_data = os.path.join(base_dir, "data_exist/test_data.pkl")
        if os.path.exists(path_valid_data):
            with open(path_valid_data, 'rb') as f:
                intr = pickle.load(f)
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')
                    user = self.user_id_mapping[line[0]]
                    if user not in intr:
                        intr[user] = [self.app_id_mapping[game] for game in line[1:]]
            with open(path_valid_data, 'wb') as f:
                pickle.dump(intr, f)
        return intr



    def read_game_genre_mapping(self, path):
        mapping = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_game_type_mapping = os.path.join(base_dir, "data_exist/game_genre_mapping.pkl")
        if os.path.exists(path_game_type_mapping):
            with open(path_game_type_mapping, 'rb') as f:
                mapping = pickle.load(f)

            return mapping

        else:
            mapping_value2id = {}
            count = 0

            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')

                    if len(line)>=2 and line[1]!= '' and line[1] not in mapping_value2id:
                        mapping_value2id[line[1]] = count
                        count += 1

                for line in tqdm(lines):
                    line = line.strip().split(',')
                    if self.app_id_mapping[line[0]] not in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]] = [line[1]]
                    elif self.app_id_mapping[line[0]] in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]].append(line[1])


                for key in tqdm(mapping):
                    mapping[key] = [mapping_value2id[x] for x in mapping[key]] 

                mapping_sort = {}
                for key in range(game_num):
                    if key not in mapping.keys():
                        mapping_sort[key] = []
                    else:
                        mapping_sort[key] = mapping[key]

                with open(path_game_type_mapping, 'wb') as f:
                    pickle.dump(mapping_sort, f)  

            return mapping



    def read_play_time_rank(self, game_path, time_path):  
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path = os.path.join(base_dir, "data_exist")
        path_tensor = path + "/tensor_user_game.pth"
        path_dic = path + "/dic_user_game.pkl"

        if os.path.exists(path_tensor) and os.path.exists(path_dic):
            tensor_user_game = torch.load(path_tensor)
            with open(path_dic, "rb") as f:
                dic_user_game = pickle.load(f)
            return tensor_user_game, dic_user_game

        else:
            ls = []
            dic_game = {}
            with open(game_path, 'r') as f_game:
                with open(time_path, 'r') as f_time:
                    lines_game = f_game.readlines()
                    lines_time = f_time.readlines()
                    for i in tqdm(range(len(lines_game))):
                        line_game = lines_game[i].strip().split(',')
                        line_time = lines_time[i].strip().split(',')
                        user = self.user_id_mapping[line_game[0]]  
                      
                        if user not in dic_game:
                            dic_game[user] = []

                       
                        idx_time_filtered = [j for j in range(1, len(line_time)) if line_time[j] != r'\N']
                        line_time_filtered = [float(line_time[j]) for j in idx_time_filtered]

                        if len(line_time_filtered) > 0:
                            ar_time = np.array(line_time_filtered)
                            time_mean = np.mean(ar_time)  
                        else:
                            continue

                        for j in range(1, len(line_game)):  
                            game = self.app_id_mapping[line_game[j]]  
                            dic_game[user].append(game)  

                            if line_time[j] == r'\N':
                                ls.append([user, game, None])
                                continue  

                            time = float(line_time[j])  
                            ls.append([user, game, time])  

            
            with open(path_dic, 'wb') as f:
                pickle.dump(dic_game, f)

            
            percentile_ls = self.generate_percentile(ls)
            for record in percentile_ls:
                if record[2] is None:  
                    record[2] = -1

            
            tensor = torch.tensor(percentile_ls, dtype=torch.float)  
            torch.save(tensor, path_tensor)
            return tensor, dic_game  

    def game_genre_inter(self, mapping):
        game_type_inter = []
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_game_genre_inter = os.path.join(base_dir, "data_exist/game_genre_inter.pkl")
        if os.path.exists(path_game_genre_inter):
            with open(path_game_genre_inter, 'rb') as f:
                game_type_inter = pickle.load(f)
        else:
            for key in tqdm(list(mapping.keys())):
                for type_key in mapping[key]:
                    game_type_inter.append([key,type_key])

            with open(path_game_genre_inter, 'wb') as f:
                pickle.dump(game_type_inter, f)

        return game_type_inter


    def read_app_info(self, path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        path_dic = os.path.join(base_dir, "data_exist/dic_app_info.pkl")
        if os.path.exists(path_dic):
            with open(path_dic, 'rb') as f:
                dic = pickle.load(f)
            return dic
        else:
            df = pd.read_csv(path, header=None)
            games = np.array(list(df.iloc[:,0])).reshape(-1,1)

            prices = np.array(list(df.iloc[:,3]))

            prices_mean = prices.mean()
            prices = prices.reshape(-1,1)


            dates = df.iloc[:,4]
            dates = np.array(list(pd.to_datetime(dates).astype('int64')))
            dates_mean = dates.mean()
            dates = (dates.astype(float)/dates.max()).reshape(-1,1)
            
            ratings = df.iloc[:,-3].replace(-1,np.nan)
            ratings_mean = ratings.mean()
            ratings = ratings.fillna(ratings_mean).values/100
            ratings = ratings.reshape(-1,1)


            app_info = np.hstack((prices,dates,ratings))
            dic = {}
            for i in range(len(games)):
                dic[self.app_id_mapping[str(games[i][0])]] = app_info[i]

            for game in self.app_id_mapping.keys():
                if game not in games:
                    dic[self.app_id_mapping[game]] = np.array([prices_mean, dates_mean, ratings_mean])


            with open(path_dic,'wb') as f:
                pickle.dump(dic,f)
            return dic
    def em_algorithm(self, game_times, game_ids=None, max_iter=100, tol=1e-6):


        game_times = np.array(game_times)
        

        zero_indices = np.where(game_times == 0)[0]
        non_zero_indices = np.where(game_times > 0)[0]
        non_zero_times = game_times[non_zero_indices]
        

        posterior_probs = np.zeros(len(game_times))
        

        if len(non_zero_times) <= 1:
            return np.zeros(len(game_times), dtype=bool), posterior_probs
        
        if len(non_zero_times) == 2:
            median = np.median(non_zero_times)
            strong_mask = np.zeros(len(game_times), dtype=bool)
            strong_mask[non_zero_indices[non_zero_times >= median]] = True
            posterior_probs[non_zero_indices[non_zero_times >= median]] = 1.0
            return strong_mask, posterior_probs
        
        
        non_zero_times = np.clip(non_zero_times, 0.001, 0.999)
        
        
        sorted_indices = np.argsort(non_zero_times)[::-1]
        cut_point = max(1, int(len(non_zero_times) * 0.4))
        
        strong_init = non_zero_times[sorted_indices[:cut_point]]
        weak_init = non_zero_times[sorted_indices[cut_point:]]
        

        mean_S = np.mean(strong_init) if len(strong_init) > 0 else 0.8
        var_S = np.var(strong_init) if len(strong_init) > 0 else 0.04
        
    
        var_S = max(var_S, 0.001)
        var_S = min(var_S, mean_S * (1 - mean_S) - 0.001)
        
  
        alpha_S = mean_S * (mean_S * (1 - mean_S) / var_S - 1)
        beta_S = (1 - mean_S) * (mean_S * (1 - mean_S) / var_S - 1)
        
     
        alpha_S = max(alpha_S, 1.1)
        beta_S = max(beta_S, 1.1)
        
     
        mean_W = np.mean(weak_init) if len(weak_init) > 0 else 0.2
        var_W = np.var(weak_init) if len(weak_init) > 0 else 0.04
        
    
        var_W = max(var_W, 0.001)
        var_W = min(var_W, mean_W * (1 - mean_W) - 0.001)
        

        alpha_W = mean_W * (mean_W * (1 - mean_W) / var_W - 1)
        beta_W = (1 - mean_W) * (mean_W * (1 - mean_W) / var_W - 1)
        
      
        alpha_W = max(alpha_W, 1.1)
        beta_W = max(beta_W, 1.1)
    
        pi = cut_point / len(non_zero_times)
        
  
        gamma = np.zeros(len(non_zero_times))
        for iteration in range(max_iter):
         
            pdf_S = stats.beta.pdf(non_zero_times, alpha_S, beta_S)
            pdf_W = stats.beta.pdf(non_zero_times, alpha_W, beta_W)
            
      
            numerator = pi * pdf_S
            denominator = pi * pdf_S + (1 - pi) * pdf_W
        
            denominator = np.maximum(denominator, 1e-10)
            gamma = numerator / denominator
            
  
            N_S = np.sum(gamma)
            N_W = len(non_zero_times) - N_S
            
     
            pi_new = N_S / len(non_zero_times)
            
        
            if N_S > 0:
         
                mean_S_new = np.sum(gamma * non_zero_times) / N_S
                var_S_new = np.sum(gamma * (non_zero_times - mean_S_new)**2) / N_S
                
         
                var_S_new = max(var_S_new, 0.001)
                var_S_new = min(var_S_new, mean_S_new * (1 - mean_S_new) - 0.001)
                
   
                alpha_S_new = mean_S_new * (mean_S_new * (1 - mean_S_new) / var_S_new - 1)
                beta_S_new = (1 - mean_S_new) * (mean_S_new * (1 - mean_S_new) / var_S_new - 1)
        
                alpha_S_new = max(alpha_S_new, 1.1)
                beta_S_new = max(beta_S_new, 1.1)
            else:
                alpha_S_new = alpha_S
                beta_S_new = beta_S
            
            if N_W > 0:
              
                mean_W_new = np.sum((1 - gamma) * non_zero_times) / N_W
                var_W_new = np.sum((1 - gamma) * (non_zero_times - mean_W_new)**2) / N_W
                
         
                var_W_new = max(var_W_new, 0.001)
                var_W_new = min(var_W_new, mean_W_new * (1 - mean_W_new) - 0.001)
            
                alpha_W_new = mean_W_new * (mean_W_new * (1 - mean_W_new) / var_W_new - 1)
                beta_W_new = (1 - mean_W_new) * (mean_W_new * (1 - mean_W_new) / var_W_new - 1)
                
                
                alpha_W_new = max(alpha_W_new, 1.1)
                beta_W_new = max(beta_W_new, 1.1)
            else:
                alpha_W_new = alpha_W
                beta_W_new = beta_W
            
           
            params_old = np.array([alpha_S, beta_S, alpha_W, beta_W, pi])
            params_new = np.array([alpha_S_new, beta_S_new, alpha_W_new, beta_W_new, pi_new])
            
            if np.all(np.abs(params_new - params_old) < tol):
                break
            
            
            alpha_S, beta_S, alpha_W, beta_W, pi = params_new
        
        
        mean_S = alpha_S / (alpha_S + beta_S)
        mean_W = alpha_W / (alpha_W + beta_W)
        
        if mean_S < mean_W:
            alpha_S, alpha_W = alpha_W, alpha_S
            beta_S, beta_W = beta_W, beta_S
            gamma = 1 - gamma
        
      
        strong_interest_threshold = 0.5
        strong_interest_mask = gamma > strong_interest_threshold
        
       
        if not np.any(strong_interest_mask) and len(non_zero_times) > 0:
            max_idx = np.argmax(non_zero_times)
            strong_interest_mask[max_idx] = True
            gamma[max_idx] = 1.0
        
     
        full_mask = np.zeros(len(game_times), dtype=bool)
        full_mask[non_zero_indices[strong_interest_mask]] = True
        
       
        posterior_probs[non_zero_indices] = gamma
        
        return full_mask, posterior_probs
    def Get_S_views(self, graph):

        save_dir = './data_exist'
        S_graph_path = os.path.join(save_dir, f"S_graph_em.bin")
        
        
        if os.path.exists(S_graph_path):
            try:
                S_graph, _ = dgl.load_graphs(S_graph_path)
                logging.info(f"Loaded existing EM-based denoised graph from {S_graph_path}")
                return S_graph[0]
            except Exception as e:
                logging.warning(f"Failed to load existing graph: {e}. Rebuilding graph...")
        
        torch.cuda.empty_cache()
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        
        S_graph = graph.clone()
        S_graph = S_graph.to('cpu')  
        

        src_play, dst_play = S_graph.edges(etype='play')
        play_edges = {}
        for i in range(len(src_play)):
            user_id = src_play[i].item()
            game_id = dst_play[i].item()
            if user_id not in play_edges:
                play_edges[user_id] = {}
            play_edges[user_id][game_id] = i
        

        user_game_times = {}
        for user_id in play_edges:
            games = list(play_edges[user_id].keys())
            indices = [play_edges[user_id][g] for g in games]
            times = S_graph.edges['play'].data['percentile'][indices].numpy()
            user_game_times[user_id] = (games, times, indices)
        

        num_edges_play = S_graph.num_edges('play')
        num_edges_played_by = S_graph.num_edges('played by')
        noisy_play = torch.zeros(num_edges_play, dtype=torch.bool)
        noisy_played_by = torch.zeros(num_edges_played_by, dtype=torch.bool)
        

        posterior_probs = torch.zeros(num_edges_play)
        
        logging.info(f"Processing {len(user_game_times)} users with EM algorithm")
        
        noisy_edges_count = 0
        

        for user_id, (games, times, indices) in tqdm(user_game_times.items()):

            if len(games) < 3:
                continue
            

            strong_interest_mask, probs = self.em_algorithm(times, games)
            

            for i, idx in enumerate(indices):
                posterior_probs[idx] = float(probs[i])
            

            noise_mask = ~strong_interest_mask
            if np.any(noise_mask):

                for i, is_noise in enumerate(noise_mask):
                    if is_noise:
                        play_idx = indices[i]
                        noisy_play[play_idx] = True
                        

                        game_id = games[i]
                        user_id_tensor = torch.tensor([user_id])
                        game_id_tensor = torch.tensor([game_id])
                        edge_id = S_graph.edge_ids(game_id_tensor, user_id_tensor, etype='played by')
                        if edge_id.numel() > 0:
                            noisy_played_by[edge_id] = True
                
                noisy_edges_count += np.sum(noise_mask)
        

        S_graph.edges['play'].data['em_posterior'] = posterior_probs
        

        reverse_posteriors = torch.zeros(num_edges_played_by)
        for i in range(num_edges_play):
            if noisy_play[i]:
                user_id = src_play[i].item()
                game_id = dst_play[i].item()
                edge_id = S_graph.edge_ids(torch.tensor([game_id]), torch.tensor([user_id]), etype='played by')
                if edge_id.numel() > 0:
                    reverse_posteriors[edge_id] = posterior_probs[i]
        
        S_graph.edges['played by'].data['em_posterior'] = reverse_posteriors
        
        
        S_graph.edges['play'].data['noisy'] = noisy_play
        S_graph.edges['played by'].data['noisy'] = noisy_played_by
        
        
        noise_edges_play = torch.nonzero(noisy_play).squeeze()
        noise_edges_played_by = torch.nonzero(noisy_played_by).squeeze()
        
        if isinstance(noise_edges_play, torch.Tensor) and noise_edges_play.numel() > 0:
            if noise_edges_play.dim() == 0:
                noise_edges_play = noise_edges_play.unsqueeze(0)
            S_graph.remove_edges(noise_edges_play, etype='play')
            logging.info(f"Removed {len(noise_edges_play)} play edges based on EM algorithm")
        
        if isinstance(noise_edges_played_by, torch.Tensor) and noise_edges_played_by.numel() > 0:
            if noise_edges_played_by.dim() == 0:
                noise_edges_played_by = noise_edges_played_by.unsqueeze(0)
            S_graph.remove_edges(noise_edges_played_by, etype='played by')
            logging.info(f"Removed {len(noise_edges_played_by)} played by edges based on EM algorithm")
        
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            dgl.save_graphs(S_graph_path, [S_graph])
            logging.info(f"Successfully saved EM-based denoised graph to {S_graph_path}")
            S_graph = S_graph.to(device)
        except Exception as e:
            logging.error(f"Failed to save EM-based denoised graph: {e}")
        finally:
            torch.cuda.empty_cache()
        
        return S_graph




    def process(self):
        logging.info("reading genre info...")
        self.genre_mapping = self.read_game_genre_mapping(self.genre_path)
        self.genre = self.game_genre_inter(self.genre_mapping)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        logging.info("reading user item play time...")
        self.user_game, self.dic_user_game = self.read_play_time_rank(self.train_game_path, self.train_time_path)

        if os.path.exists(os.path.join(base_dir, "data_exist/graph.bin")):
            graph,_ = dgl.load_graphs(os.path.join(base_dir, "data_exist/graph.bin"))
            graph = graph[0]
            self.graph = graph
            
        else:
            graph_data = {
                ('game', 'genre', 'type'): (torch.tensor(self.genre)[:,0], torch.tensor(self.genre)[:,1]),

                ('type', 'genred', 'game'): (torch.tensor(self.genre)[:,1], torch.tensor(self.genre)[:,0]),

                ('user', 'play', 'game'): (self.user_game[:, 0].long(), self.user_game[:, 1].long()),

                ('game', 'played by', 'user'): (self.user_game[:, 1].long(), self.user_game[:, 0].long())
            }
            graph = dgl.heterograph(graph_data)
            graph.edges['play'].data['time'] = self.user_game[:, 2].to(torch.float32) 
            graph.edges['played by'].data['time'] = self.user_game[:, 2].to(torch.float32)
      
            graph.edges['play'].data['percentile'] = self.user_game[:, 3]
            graph.edges['played by'].data['percentile'] = self.user_game[:, 3]

            self.graph = graph
            dgl.save_graphs(os.path.join(base_dir, "data_exist/graph.bin"),[graph])




    def __getitem__(self, i):
        pass

    def __len__(self):
        pass


    def ceshi(self):
        print(self.genre_mapping)

