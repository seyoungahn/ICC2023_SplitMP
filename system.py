import numpy as np
from numpy.random import rand
from node import UE, AP

class System(object):
    def __init__(self, params):
        self.params = params
        self.UEs = []
        for i in range(params['n_nodes']):
            self.UEs.append(UE(i, params))

    def deploy_UEs