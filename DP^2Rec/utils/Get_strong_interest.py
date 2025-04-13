import torch
import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from dataloader_steam import Dataloader_steam_filtered
from parser import parse_args

def main():

    args = parse_args()
    

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
    

    logging.info("Generating EM-based denoised graph...")
    denoised_graph = dataloader.Get_S_views(dataloader.graph)
    
    logging.info("EM denoising completed successfully!")
    

    logging.info(f"Original graph: {dataloader.graph.number_of_edges('play')} play edges")
    logging.info(f"Denoised graph: {denoised_graph.number_of_edges('play')} play edges")
    logging.info(f"Removed {dataloader.graph.number_of_edges('play') - denoised_graph.number_of_edges('play')} play edges ({(dataloader.graph.number_of_edges('play') - denoised_graph.number_of_edges('play'))/dataloader.graph.number_of_edges('play')*100:.2f}%)")
    
    return denoised_graph

if __name__ == "__main__":
    denoised_graph = main()
    print("EM-based denoised graph generation completed!")