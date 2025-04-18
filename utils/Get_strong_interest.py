import torch
import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import dgl
from dataloader_steam import Dataloader_steam_filtered
from parser import parse_args

def main():

    args = parse_args()
    S_graph_path = "data_exist/S_graph.bin"
    if os.path.exists(S_graph_path):
        logging.info(f"Output file {S_graph_path} already exists, skipping entire processing pipeline")
        # Load the existing graph to return
        logging.info(f"Loading existing graph from {S_graph_path}")
        graph, _ = dgl.load_graphs(S_graph_path)
        return graph[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    path = "./steam_data"
    user_id_path = path + '/users.txt'
    app_id_path = path + '/app_id.txt'
    genre_path = path + '/Games_Genres.txt'
    

    logging.info("Initializing data loader...")
    dataloader = Dataloader_steam_filtered(args,
                                      path,
                                      user_id_path,
                                      app_id_path,
                                      genre_path,                       
                                      device)
    

    logging.info("Generating EM-based strong interest graph...")
    denoised_graph = dataloader.Get_S_views(dataloader.graph)
    
    logging.info("EM completed successfully!")
    

    logging.info(f"Original graph: {dataloader.graph.number_of_edges('play')} play edges")
    logging.info(f"Denoised graph: {denoised_graph.number_of_edges('play')} play edges")
    logging.info(f"Removed {dataloader.graph.number_of_edges('play') - denoised_graph.number_of_edges('play')} play edges ({(dataloader.graph.number_of_edges('play') - denoised_graph.number_of_edges('play'))/dataloader.graph.number_of_edges('play')*100:.2f}%)")
    
    return denoised_graph

if __name__ == "__main__":
    denoised_graph = main()
    print("EM-based strong interest graph generation completed!")