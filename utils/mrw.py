import os
import sys
import numpy as np
import torch
import dgl
from tqdm import tqdm
import pickle
import logging
from collections import defaultdict, Counter
import random
import multiprocessing as mp
import threading
import os
import time
import sys
import csv
from parser import parse_args
from dataloader_steam import Dataloader_steam_filtered
import multiprocessing as mp
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPUs: {torch.cuda.device_count()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    return device


DEVICE = get_device()


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.cudnn.benchmark = True

def min_max_normalize(values):
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [1.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def roulette_wheel_selection(probabilities):
    r = random.random()
    cumulative_prob = 0
    for i, prob in enumerate(probabilities):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return i
    return len(probabilities) - 1  

def prepare_embeddings_gpu(embeddings_dict):
    ids = []
    embeddings = []
    
    for game_id, embedding in embeddings_dict.items():
        ids.append(game_id)
        embeddings.append(embedding)
    
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
    id_to_idx = {game_id: i for i, game_id in enumerate(ids)}
    idx_to_id = {i: game_id for i, game_id in enumerate(ids)}
    
    return embeddings_tensor, id_to_idx, idx_to_id

def precompute_genre_similar_games(embeddings_tensor, id_to_idx, game_to_genres, genre_to_games):
    precomputed_similar_games = {}
    all_game_ids = list(id_to_idx.keys())
    for game_id in tqdm(all_game_ids):
        if game_id not in id_to_idx:
            continue
        current_idx = id_to_idx[game_id]
        
        all_genres = list(genre_to_games.keys())
        
        game_similar_games = []
        
        for genre_id in all_genres:
            genre_games = genre_to_games[genre_id]
            
            genre_games = [g for g in genre_games if g != game_id and g in id_to_idx]
            
            if not genre_games:
                continue
            
            genre_indices = [id_to_idx[g] for g in genre_games]
            
            current_embedding = embeddings_tensor[current_idx].unsqueeze(0)
            genre_embeddings = embeddings_tensor[genre_indices]
            
            modal_similarities = torch.mm(current_embedding, genre_embeddings.t()).squeeze(0)
            

            if len(modal_similarities) > 0:
                best_idx = torch.argmax(modal_similarities).item()
                best_game_id = genre_games[best_idx]
                best_similarity =modal_similarities[best_idx].item()
                

                game_similar_games.append((best_game_id, best_similarity, genre_id))
        

        precomputed_similar_games[game_id] = game_similar_games
    
    logger.info(f"Successfully precomputed {len(precomputed_similar_games)} games' similar games in each category.")
    return precomputed_similar_games

def reverse_mapping(mapping):
    return {v: k for k, v in mapping.items()}

def load_game_embeddings(embedding_folder):
    embeddings = {}
    files = os.listdir(embedding_folder)
    for file in tqdm(files):
        if file.endswith('.npy'):
            game_id = file.split('.')[0]
            embedding = np.load(os.path.join(embedding_folder, file))
            embeddings[game_id] = embedding.flatten() 

    logger.info(f"Successfully loaded embeddings for {len(embeddings)} games.")
    return embeddings

def load_user_game_interactions(DataLoader):
    user_games = {}
    user_game_times = {}
    
    dic_user_game = DataLoader.dic_user_game
    tensor_user_game = DataLoader.user_game

    for i in range(tensor_user_game.shape[0]):
        user_id = int(tensor_user_game[i, 0].item())
        game_id = int(tensor_user_game[i, 1].item())
        time_percentile = tensor_user_game[i, 3].item()  

        if user_id not in user_game_times:
            user_game_times[user_id] = {}

        user_game_times[user_id][game_id] = time_percentile

    for user_id, games in dic_user_game.items():
        user_games[user_id] = set(games)

    logger.info(f"Successfully loaded game interaction information for {len(user_games)} users.")
        
    return user_games, user_game_times

def get_S_user_games(S_graph, user_id):

    src, dst = S_graph.out_edges(user_id, etype='play')
    if len(src) == 0:
        return []
    
    S_games = dst.tolist()
    return S_games


def find_similar_games_per_genre(current_game_id, precomputed_similar_games):

    return precomputed_similar_games.get(current_game_id, [])

def load_game_time_similarity(file_path, app_id_forward):
    game_time_sim = {}

    int_to_numeric_id = {}
    for orig_id, numeric_id in app_id_forward.items():
        try:
            int_id = int(orig_id)
            int_to_numeric_id[int_id] = numeric_id
        except ValueError:
            pass
    

        
    count = 0
    matched = 0
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        
        for row in tqdm(reader, desc="Processing similarity data"):
            if len(row) < 3:
                continue

                
            try:
                game1_int = int(row[0])
                game2_int = int(row[1])
                similarity = float(row[2])
                
                count += 1
                
                if game1_int in int_to_numeric_id and game2_int in int_to_numeric_id:
                    game1 = int_to_numeric_id[game1_int]
                    game2 = int_to_numeric_id[game2_int]
                    
                    if game1 not in game_time_sim:
                        game_time_sim[game1] = {}
                    if game2 not in game_time_sim:
                        game_time_sim[game2] = {}
                        
                    game_time_sim[game1][game2] = similarity
                    game_time_sim[game2][game1] = similarity
                    
                    matched += 1
            except (ValueError, IndexError):
                continue
                
            if count % 1000000 == 0:
                logger.info(f"Processed {count} game pairs, successfully matched {matched} pairs")
    
    logger.info(f"Total processed {count} game pairs, successfully matched {matched} pairs")
    logger.info(f"Successfully loaded game time similarity data, covering {len(game_time_sim)} games")
    

    
    return game_time_sim

def process_user(user_id, dn_graph, user_games, user_game_times, 
                precomputed_similar_games, game_to_genres, game_time_sim, max_per_genre):

    original_games = user_games.get(user_id, set())
    

    denoised_games = get_S_user_games(dn_graph, user_id)
    
    if not denoised_games:
        return [], []
    
    game_times = {}
    for game_id in denoised_games:
        if user_id in user_game_times and game_id in user_game_times[user_id]:
            game_times[game_id] = user_game_times[user_id][game_id]
    
    if not game_times:
        return [], []
    

    genre_counts = defaultdict(int)
    
    added_games = set()
    
    user_to_game_edges = []
    user_game_weights = []
    

    explored_initial_nodes = set()  
    exploration_results = {}  
    
    game_level_info = {}  
    
    while True:
        if not game_times:
            break
            
        times = list(game_times.values())
        game_ids = list(game_times.keys())
        total_time = sum(times)
        
        if total_time == 0:
            probs = [1/len(times)] * len(times)
        else:
            probs = [t/total_time for t in times]
            
        selected_idx = roulette_wheel_selection(probs)
        initial_game_id = game_ids[selected_idx]
        
        if initial_game_id in explored_initial_nodes:
            previous_added = exploration_results.get(initial_game_id, [])
            
            if not previous_added:
                continue
                
            start_game_id = random.choice(previous_added)
        else:
            start_game_id = initial_game_id
            game_level_info[initial_game_id] = (1, [initial_game_id], 1.0)
            exploration_results[initial_game_id] = []
            explored_initial_nodes.add(initial_game_id)
        
        current_level, current_path, current_accumulated_sim = game_level_info.get(start_game_id, (1, [start_game_id], 1.0))
        
        similar_games = find_similar_games_per_genre(start_game_id, precomputed_similar_games)
        
        filtered_similar_games = []
        for game_id, modal_sim, genre_id in similar_games:
            if game_id not in original_games and game_id not in added_games:

                time_sim = 0.0
                if start_game_id in game_time_sim and game_id in game_time_sim[start_game_id]:
                    time_sim = game_time_sim[start_game_id][game_id]
                filtered_similar_games.append((game_id, modal_sim, time_sim, genre_id))
        
        if not filtered_similar_games:
            continue
    
        candidates = []
        sim_values = []
        time_sim_values = []
        category_values = []
        
        for game_id, modal_sim, time_sim, genre_id in filtered_similar_games:
            current_count = genre_counts[genre_id]
            
            
            if current_count >= max_per_genre:
                continue
            
            remaining_ratio = (max_per_genre - current_count) / max_per_genre
            
            candidates.append((game_id, genre_id))
            sim_values.append(modal_sim)
            time_sim_values.append(time_sim)
            category_values.append(remaining_ratio)
        
        if not candidates:
            continue
        
        norm_sim_values = min_max_normalize(sim_values)
        norm_time_sim_values = min_max_normalize(time_sim_values)
        norm_category_values = min_max_normalize(category_values)
        
        scores = []
        for i in range(len(candidates)):
            score = (norm_sim_values[i] + norm_time_sim_values[i] + norm_category_values[i]) / 3
            scores.append(score)
        
        total_score = sum(scores)
        if total_score == 0:
            probs = [1/len(scores)] * len(scores)
        else:
            probs = [s/total_score for s in scores]
        
        selected_idx = roulette_wheel_selection(probs)
        selected_game_id, selected_genre_id = candidates[selected_idx]
        
        user_to_game_edges.append((user_id, selected_game_id))
        
        selected_genre_count = len(game_to_genres.get(selected_game_id, []))

        
        modal_sim = sim_values[selected_idx]
        
        new_level = current_level + 1
        new_path = current_path + [selected_game_id]
        new_accumulated_sim = current_accumulated_sim * modal_sim
        

        game_level_info[selected_game_id] = (new_level, new_path, new_accumulated_sim)
        
        if new_level == 1: 
            user_game_time = user_game_times[user_id].get(initial_game_id, 0)
            edge_weight = user_game_time
        elif new_level == 2:  
            initial_game_id = new_path[0]
            user_game_time = user_game_times[user_id].get(initial_game_id, 0)
            edge_weight = user_game_time * modal_sim
        else:  
            initial_game_id = new_path[0]
            user_game_time = user_game_times[user_id].get(initial_game_id, 0)
            edge_weight = user_game_time * new_accumulated_sim
        
        user_game_weights.append(edge_weight)
        
        added_games.add(selected_game_id)
        

        for genre_id in game_to_genres.get(selected_game_id, []):
            genre_counts[genre_id] += 1
        
        if initial_game_id in exploration_results:
            exploration_results[initial_game_id].append(selected_game_id)
            
        all_full = True
        for genre_id in genre_counts.keys():
            if genre_counts[genre_id] < max_per_genre:
                all_full = False
                break
                
        if all_full:
            break
    
    return user_to_game_edges, user_game_weights

def batch_process_users(user_ids, dn_graph, user_games, user_game_times, 
                     precomputed_similar_games, game_to_genres, game_time_sim, max_per_genre, batch_size=1024):

    all_edges = []
    all_weights = []
    
    num_batches = (len(user_ids) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(user_ids))
        batch_user_ids = user_ids[start_idx:end_idx]
        
        batch_results = []
        for user_id in batch_user_ids:

            start_time = time.time()
            

            edges, weights = process_user(
                user_id, 
                dn_graph, 
                user_games, 
                user_game_times, 
                precomputed_similar_games,
                game_to_genres,
                game_time_sim,
                max_per_genre
            )
            

            
            batch_results.append((edges, weights))

        

        for edges, weights in batch_results:
            all_edges.extend(edges)
            all_weights.extend(weights)
        

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_edges, all_weights

def main():
    set_random_seed(2025)

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    args = parse_args()
    path = "steam_data"
    user_id_path = path + '/users.txt'
    app_id_path = path + '/app_id.txt'
    genres_path = path + '/Games_Genres.txt'
    embedding_folder = "steam_data/modal_embeddings"   
    output_graph_path = "data_exist/mrw.bin" 
    S_graph_path ="data_exist/S_graph.bin"
    weights_path = "data_exist/weights_mrw.pth"
    if os.path.exists(output_graph_path) and os.path.exists(weights_path):
        logger.info(f"Output files {output_graph_path} and {weights_path} already exist, skipping entire processing pipeline")
        return
    logger.info("Loading data...")
    DataLoader = Dataloader_steam_filtered(args, path, user_id_path, app_id_path, genres_path)
    

    logger.info("Creating ID mappings...")
    app_id_reverse = reverse_mapping(DataLoader.app_id_mapping)
    app_id_forward = DataLoader.app_id_mapping  
    user_id_reverse = reverse_mapping(DataLoader.user_id_mapping)
    user_id_forward = DataLoader.user_id_mapping  

    game_time_sim_path = "data_exist/game_similarity.csv"
    game_time_sim = load_game_time_similarity(game_time_sim_path, app_id_forward)

    logger.info("Loading genre mappings...")
    genre_id_to_name = {}
    name_to_genre_id = {}
    

    with open(genres_path, 'r') as f:
        genre_set = set()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[1]:
                genre_set.add(parts[1])
        
        for i, genre in enumerate(sorted(genre_set)):
            genre_id_to_name[i] = genre
            name_to_genre_id[genre] = i

    

    game_to_genres = {}
    game_to_genre_names = {}
    
    for game_id, genre_ids in DataLoader.genre_mapping.items():
        game_to_genres[game_id] = genre_ids
        game_to_genre_names[game_id] = [genre_id_to_name.get(genre_id, f"类别_{genre_id}") for genre_id in genre_ids]
    

    genre_to_games = defaultdict(list)
    for game_id, genre_ids in game_to_genres.items():
        for genre_id in genre_ids:
            genre_to_games[genre_id].append(game_id)
    

    

    original_game_embeddings = load_game_embeddings(embedding_folder)
    

    embeddings_mapped = {}
    for original_id, embedding in original_game_embeddings.items():
        if original_id in app_id_forward:
            numeric_id = app_id_forward[original_id]
            embeddings_mapped[numeric_id] = embedding
    
    embeddings_tensor, id_to_idx, idx_to_id = prepare_embeddings_gpu(embeddings_mapped)

    precomputed_similar_games = precompute_genre_similar_games(embeddings_tensor, id_to_idx, game_to_genres, genre_to_games)

    
    user_games, user_game_times = load_user_game_interactions(DataLoader)
    

    dn_graph, _ = dgl.load_graphs(S_graph_path)
    dn_graph = dn_graph[0]

    if torch.cuda.is_available():
        dn_graph = dn_graph.to(DEVICE)

    
    test_users = list(DataLoader.test_data.keys())
    
    batch_size = 1
    all_edges, all_weights = batch_process_users(
        test_users, 
        dn_graph, 
        user_games, 
        user_game_times, 
        precomputed_similar_games,
        game_to_genres,
        game_time_sim,
        args.max_per_genre,  
        batch_size
    )
    

        
    src_nodes = torch.tensor([edge[0] for edge in all_edges], dtype=torch.int64)
    dst_nodes = torch.tensor([edge[1] for edge in all_edges], dtype=torch.int64)
    

    edge_weights = torch.tensor(all_weights, dtype=torch.float32)
    
  
    actual_users = torch.unique(src_nodes).shape[0]
    actual_games = torch.unique(dst_nodes).shape[0]
    
    logger.info(f"Number of users with actual connections: {actual_users}")
    logger.info(f"Number of games with actual connections: {actual_games}")

    
 
    graph_data = {
        ('user', 'plays', 'game'): (src_nodes, dst_nodes),
        ('game', 'played_by', 'user'): (dst_nodes, src_nodes)
    }
    
    num_nodes_dict = {
        'user': DataLoader.graph.number_of_nodes('user'),
        'game': DataLoader.graph.number_of_nodes('game')
    }
    
    diversity_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    

    diversity_graph.edges['plays'].data['weight'] = edge_weights
    diversity_graph.edges['played_by'].data['weight'] = edge_weights
    
 
    logger.info(f"Saving diversity exploration graph to {output_graph_path}")

    dgl.save_graphs(output_graph_path, [diversity_graph])
    


    torch.save(edge_weights, weights_path)
    logger.info(f"Edge weights saved to {weights_path}")

    
    num_users = diversity_graph.number_of_nodes('user')
    num_games = diversity_graph.number_of_nodes('game')
    num_edges = diversity_graph.number_of_edges('plays')

    logger.info("Diversity exploration graph statistics:")
    logger.info(f"  - Total number of users: {num_users}")
    logger.info(f"  - Total number of games: {num_games}")
    logger.info(f"  - Number of edges: {num_edges}")

    avg_connections = num_edges / actual_users if actual_users > 0 else 0
    logger.info(f"  - Average number of games per active user: {avg_connections:.2f}")

    user_genre_coverage = defaultdict(set)

    for i in range(len(all_edges)):
        user_id, game_id = all_edges[i]
        for genre_id in game_to_genres.get(game_id, []):
            user_genre_coverage[user_id].add(genre_id)

    coverage_counts = [len(genres) for user_id, genres in user_genre_coverage.items()]

    if coverage_counts:
        avg_genre_coverage = sum(coverage_counts) / len(coverage_counts)
        max_genre_coverage = max(coverage_counts)
        min_genre_coverage = min(coverage_counts)

        logger.info(f"  - Average number of genres covered per user: {avg_genre_coverage:.2f}")
        logger.info(f"  - Maximum genre coverage: {max_genre_coverage}")
        logger.info(f"  - Minimum genre coverage: {min_genre_coverage}")

    logger.info("Calculating connection statistics per genre...")
    genre_connected_users = defaultdict(set)
    genre_interaction_count = defaultdict(int)

    for i in range(len(all_edges)):
        user_id, game_id = all_edges[i]
        for genre_id in game_to_genres.get(game_id, []):
            genre_connected_users[genre_id].add(user_id)
            genre_interaction_count[genre_id] += 1

    logger.info("Connection statistics per game genre:")
    sorted_genres = sorted([(genre_id, genre_interaction_count[genre_id], len(genre_connected_users[genre_id])) 
                            for genre_id in genre_interaction_count.keys()], 
                        key=lambda x: x[1], reverse=True)

    for genre_id, interaction_count, user_count in sorted_genres:
        genre_name = genre_id_to_name.get(genre_id, f"Genre_{genre_id}")
        total_genre_games = len(genre_to_games[genre_id])
        avg_interactions_per_user = interaction_count / user_count if user_count > 0 else 0
        logger.info(f"  - {genre_name}: {interaction_count} total interactions, {user_count} users, "
                    f"avg {avg_interactions_per_user:.2f} games per user in this genre "
                    f"(total {total_genre_games} games in this genre)")

    logger.info("Diversity exploration graph construction completed!")



if __name__ == "__main__":
    main()