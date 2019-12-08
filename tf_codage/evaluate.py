import numpy as np

def print_support(encoded):
    """Print support given binarised labels"""
    
    encoded = np.asarray(encoded)
    support = encoded.sum(0)
    
    print("    n° classes")
    print("")
    print("all {:>10d}".format(encoded.shape[1]))
    print("> 0 ex {:>7d}".format((support>0).sum()))
    print("> 10 ex {:>6d}".format((support>10).sum()))
    print("")
    print("       support")
    print(" (n° examples)")
    print("")
    print("sum {:>10d}".format(np.sum(support)))
    print("median {:>7.1f}".format(np.median(support)))
    print("max {:>10d}".format(np.max(support)))
    print("min {:>10d}".format(np.min(support)))
    
def get_encoded_array(dataset):
    """Take TF dataset and return an array of binarised labels"""
    real_encoded = np.vstack([s[1].numpy() for s in dataset])

    return real_encoded