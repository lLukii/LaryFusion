import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from config import Config
from dataset import (
    load_wavs, 
    batch_inputs, 
    cross_validation
)
import warnings
warnings.filterwarnings("ignore")
config = Config()

def train_loop(dataset, test_dataset, model, optimizer, loss_fn, log_interval=10,          
               include_demographic=False): 
    # training
    training_steps = 0
    with tqdm(desc="Training in progress....") as pbar:
        for x, y in dataset:
            x, y = x.to(config.device), y.to(config.device).long() # Convert labels to long
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            loss_fn.step()
            training_steps += 1
            if training_steps % log_interval == 0: 
                print("Loss value:", loss.item())


        test_size = (len(test_dataset) - 1) * config.batch_size + len(test_dataset[-1])
        correct = 0
        for x, y in test_dataset: 
            x, y = x.to(config.device), y.to(config.device).long() 
            pred = model(x)
            correct += (pred.round() == y).sum().item()
            loss = loss_fn(pred, y)

        print("Accuracy:", correct / test_size)


def visualize_data(): 
    pass

def parse_args():
    parser = ArgumentParser(description="test")
    parser.add_argument("--aft", type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__': 
    dataset = load_wavs(include_demographics=True)
    train_data, test_data = cross_validation(dataset)
    train_loader = batch_inputs(train_data, oversample=True)
    test_data = batch_inputs(test_data)

    print(train_loader)
