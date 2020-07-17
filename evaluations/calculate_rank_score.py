import numpy as np

def load_idx(path):
    with open(path, 'r') as f:
        all_idx = f.readlines()
        all_idx = [[int(a) for a in s.strip()] for s in all_idx]
    return np.array(all_idx)


def np_rerank2d(base, idx):
    new_lines = []
    X,Y = base.shape
    for x in range(X):
        new_line = [base[x][y-1] for y in idx[x]]
        new_lines.append(new_line)
    return np.array(new_lines)

def calculate_rank_score(all_relate):
    all_relate = all_relate.reshape([-1, 18, 4])
    all_relate_ground = [np_rerank2d(all_truth, relate) for relate in all_relate]
    ground_relate_dicts = {x: [] for x in range(4)}
    for p in all_relate_ground:
        for l in p:
            for key, value in enumerate(l):
                ground_relate_dicts[key].append(value)
    for key, value in ground_relate_dicts.items():
        print(key, np.average(np.array(value)))

all_truth = load_idx('truth.txt')
all_relate = load_idx('relatness.txt')
all_reality = load_idx('reality.txt')

calculate_rank_score(all_relate)
calculate_rank_score(all_reality)
