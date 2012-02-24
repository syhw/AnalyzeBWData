import sys, os, pickle, copy, itertools, functools
from common import data_tools
from common import unit_types
try:
    import numpy as np
except:
    print "you need numpy"
    sys.exit(-1)

def evaluate_pop(d):
    r = {}
    for (k,v) in d.iteritems():
        s = 0
        for (kk, vv) in v.iteritems():
            s += unit_types.unit_double_pop.get(kk,0)*vv
        r[k] = s
    return r

def extract_armies_battles(f):
    """ take a file and parse it for attacks, returning a list of attacks
    with for each time of the beginning of the attack,
    the (max) armies of each player in form of a dict and
    the remaining units at the end in form of a dict and
    the number of workers lost for each player:
        [(attack_time, {plid:{uid:nb}}, {plid:{uid:nb}}, {plid:wrks_lost})] """
    attacks = []
    for line in f:
        if 'IsAttacked' in line:
            tmp = data_tools.parse_dicts(line, lambda x: int(x))
            attacks.append((int(line.split(',')[0]),tmp[0], tmp[1], tmp[2]))
    return attacks

def format_battle_for_regr(players_races, armies_battle):
    """ take an "extract_armies_battles" formatted battle data and make it
    ready for regression with P > T > Z in order, then on player id order,
    and player1_score - player2_score """
    p = []
    for k in armies_battle[1].iterkeys():
        p.append((k,players_races[k]))
    assert(len(p) == 2)
    p1 = p[0][0]
    p2 = p[1][0]
    if p[0][1] == 'T' and p[1][1] == 'P':
        p1 = p[1][0]
        p2 = p[0][0]
    elif p[0][1] == 'Z':
        p1 = p[1][0]
        p2 = p[0][0]
    tmp = []
    for unit in unit_types.by_race.military[players_races[p1]]:
        tmp.append(armies_battle[1][p1].get(unit, 0))
    for unit in unit_types.by_race.static_defense[players_races[p1]]:
        tmp.append(armies_battle[1][p1].get(unit, 0))
    tmp.append(armies_battle[1][p1].get(unit_types.by_race.workers[
        players_races[p1]], 0))
    for unit in unit_types.by_race.military[players_races[p2]]:
        tmp.append(armies_battle[1][p2].get(unit, 0))
    for unit in unit_types.by_race.static_defense[players_races[p2]]:
        tmp.append(armies_battle[1][p2].get(unit, 0))
    tmp.append(armies_battle[1][p2].get(unit_types.by_race.workers[
        players_races[p2]], 0))
    scores = evaluate_pop(armies_battle[2])
    wrk_scores = armies_battle[3][p1]*2 - armies_battle[3][p1]*2
    tmp.append(scores[p1] - scores[p2] + wrk_scores)
    return tmp

f = sys.stdin
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if os.path.exists('raw.blob') and os.path.exists('fscaled.blob'):
            raw = pickle.load(open('raw.blob', 'r'))
            fscaled = pickle.load(open('fscaled.blob', 'r'))
        else:
            armies_battles_for_regr = []
            # [[P1units],[P2units],battle_result]
            if sys.argv[1] == '-d': # -d for directory
                import glob
                for fname in glob.iglob(sys.argv[2] + '/*.rgd'):
                    f = open(fname)
                    battles = map(functools.partial(format_battle_for_regr,
                            data_tools.players_races(f)),
                            extract_armies_battles(f))
                    armies_battles_for_regr.extend(battles)
            else:
                for arg in sys.argv[1:]:
                    f = open(arg)
                    battles = map(functools.partial(format_battle_for_regr,
                            data_tools.players_races(f)),
                            extract_armies_battles(f))
                    armies_battles_for_regr.extend(battles)
            armies_battles_regr_raw = np.array(armies_battles_for_regr, np.float32)
            armies_battles_regr_fscaled = data_tools.features_scaling(armies_battles_regr_raw)
            from sklearn import linear_model
            X = armies_battles_regr_fscaled[:,:-1]
            Y = armies_battles_regr_fscaled[:,-1]
            clf = linear_model.LinearRegression()
            print clf.fit(X, Y)
            print clf.score(X, Y)
            clf = linear_model.BayesianRidge()
            print clf.fit(X, Y)
            print clf.score(X, Y)
            clf = linear_model.Lasso(alpha = 0.1)
            print clf.fit(X, Y)
            print clf.score(X, Y)



