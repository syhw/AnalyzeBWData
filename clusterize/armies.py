import sys, os, pickle, copy, itertools
from common import data_tools

def extract_armies(f):
    """ take a file and parse it for attacks, returning a list of attacks
    with for each the (max) armies of each player in form of a dict and
    the remaining units at the end in form of a dict """
    attacks = []
    for line in f:
        if 'IsAttacked' in line:
            tmp = data_tools.parse_dicts(line, lambda x: int(x))
            attacks.append(tmp)
    return attacks

f = sys.stdin
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if os.path.exists('raw.blob') and os.path.exists('fscaled.blob'):
                raw = pickle.load(open('raw.blob', 'r'))
                fscaled = pickle.load(open('fscaled.blob', 'r'))
        else:
            armies_list = []
            if sys.argv[1] == '-d': # -d for directory
                import glob
                for fname in glob.iglob(sys.argv[2] + '/*.rgd'):
                    f = open(fname)
                    armies_list = extract_armies(f)
            else:
                for arg in sys.argv[1:]:
                    f = open(arg)
                    armies_list = extract_armies(f)
            print armies_list

    #data_tools.features_scaling()

