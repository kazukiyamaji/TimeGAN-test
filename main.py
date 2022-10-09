from Dataset.get_data import get_data
from trainer import Trainer
from models.TimeGAN import *
import argparse
from utils import parse
#datas=get_data(dataset_name="stock",seq_len=4)
def main(args):
    datas,times,max_seq=get_data(args)
    args.max_seq=max_seq  
    timegan=TimeGAN(args)
    trainer=Trainer(args,timegan,datas,times,max_seq)
    trainer.train()
    #mode=["AutoEncoder" , "Surpervise" , "Discriminator" , "Generator" , "generate"]
    

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser = parse(parser)
    args=parser.parse_args()

    main(args)
    

