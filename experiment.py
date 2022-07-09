import torch
import data_utils
from system import System

class Experiment:
    def __init__(self, exp_name, params):
        self.params = params
        self.exp_name = exp_name
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print("DEVICE: {}".format(self.device))

        self.data_gen = data_utils.MobilityDataGenerator()

    def set_system(self):
        self.system = System(self.params)
        ## TODO: (1) Distributing APs and CPUs of cell-free massive MIMO systems

        ## TODO: (2) Distributing UEs in the systems

    def __iter__(self):
        while True:
            measurements = {}
            ## TODO: (3-1) System measurements - next waypoint predictions

            ## TODO: (3-2) System measurements - channel predictions

            ## TODO: (3-3) UE movements based on RWP mobility model

            yield measurements