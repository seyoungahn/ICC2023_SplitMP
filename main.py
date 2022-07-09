import os
import utils

if __name__ == "__main__":
    ## Hyperparameter settings
    json_path = os.path.join("params.json")
    assert os.path.isfile(json_path), "No configuration file (JSON) found at {}".format(json_path)
    params = utils.Params(json_path)

    print(params.models[0]['model'])