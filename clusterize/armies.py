import sys, os, pickle, copy, itertools, functools
from common import data_tools
from common import unit_types
from common import attack_tools
try:
    import numpy as np
except:
    print "you need numpy"
    sys.exit(-1)

SCORES_REGRESSION = False
MIN_POP_ENGAGED = 6 # 12 zerglings, 6 marines, 3 zealots
MAX_FORCES_RATIO = 1.5 # max differences between engaged forces

def evaluate_pop(d):
    r = {}
    for (k,v) in d.iteritems():
        s = 0
        for (kk, vv) in v.iteritems():
            s += unit_types.unit_double_pop.get(kk,0)*vv
        r[k] = s
    return r

def score_units(d):
    r = {}
    for k,v in d.iteritems():
        r[k] = 0
        for unit, numbers in v.iteritems():
            r[k] += unit_types.score_unit(unit)*numbers
    return r

def to_ratio(d):
    r = {}
    for k,v in d.iteritems():
        s = sum(v.itervalues())
        tmp = {}
        for unit, numbers in v.iteritems():
            tmp[unit] = 1.0*numbers/s
        r[k] = tmp
    return r

def extract_armies_battles(f):
    """ take a file and parse it for attacks, returning a list of attacks
    with for each time of the beginning of the attack,
    the (max) armies of each player in form of a dict and
    the remaining units at the end in form of a dict and
    the number of workers lost for each player:
        [(attack_time, {plid:{uid:nb}}, {plid:{uid:nb}}, {plid:wrks_lost})] """ 
    attacks = []
    obs = attack_tools.Observers()
    for line in f:
        obs.detect_observers(line)
        if 'IsAttacked' in line:
            tmp = data_tools.parse_dicts(line, lambda x: int(x))
            # sometimes the observers are detected in the fight (their SCVs)
            tmp = obs.heuristics_remove_observers(tmp)
            attacks.append((int(line.split(',')[0]),tmp[0], tmp[1], tmp[2]))
    return attacks

def format_battle_init(players_races, army):
    p = []
    for k in army.iterkeys():
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
    return p1, p2

def append_units_numbers(l, player, players_races, army):
    for unit in unit_types.by_race.military[players_races[player]]:
        l.append(army[player].get(unit, 0))
    for unit in unit_types.by_race.static_defense[players_races[player]]:
        l.append(army[player].get(unit, 0))
    l.append(army[player].get(unit_types.by_race.workers[
        players_races[player]], 0))

def format_battle_for_regr(players_races, armies_battle):
    """ take an "extract_armies_battles" formatted battle data and make it
    ready for regression with P > T > Z in order, then on player id order,
    and player1_score - player2_score """
    p1, p2 = format_battle_init(players_races, armies_battle[1])
    tmp = []
    append_units_numbers(tmp, p1, players_races, armies_battle[1])
    append_units_numbers(tmp, p2, players_races, armies_battle[1])
    #scores = evaluate_pop(armies_battle[2])
    scores = score_units(armies_battle[2])
    wrk_scores = armies_battle[3][p1]*2 - armies_battle[3][p1]*2
    tmp.append(scores[p1] - scores[p2] + wrk_scores)
    return tmp

def format_battle_for_clust_adv(players_races, armies_battle):
    """ take an "extract_armies_battles" formatted battle data and make it
    ready for clustering (order P > T > Z) by considering only battles 
    which were relevant (comparable units scores for both parties), 
    returning 2 vectors of units ratio (of total army) composition
    per unit types and the final scores of both players"""
    p1, p2 = format_battle_init(players_races, armies_battle[1])
    pop_max = evaluate_pop(armies_battle[1])
    #pop_after = evaluate_pop(armies_battle[2])
    score_before = score_units(armies_battle[1])
    if pop_max[p1] > MIN_POP_ENGAGED*2 and pop_max[p2] > MIN_POP_ENGAGED*2 and score_before[p1] < MAX_FORCES_RATIO*score_before[p2] and score_before[p2] < MAX_FORCES_RATIO*score_before[p1]:
        compo = to_ratio(armies_battle[1])
        score_after = score_units(armies_battle[2])
        return compo[p1], compo[p2], score_after[p1], score_after[p2]
    else:
        return [], [], [], []

def format_battle_for_clust(players_races, armies_battle):
    """ take an "extract_armies_battles" formatted battle data and make it
    ready for clustering (ex: 'Protoss Dragoon': 12, 'Protoss Zealot': 6)
    returning 1 dict of vectors of units numbers per unit types, per race """
    r = {'P': [], 'T': [], 'Z': []}
    for k,v in players_races.iteritems():
        if k in armies_battle[1]: # armies_battle[1] is "max units engaged"
            r[v].append(copy.deepcopy(armies_battle[1][k]))
    return r

f = sys.stdin
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if os.path.exists('raw.blob') and os.path.exists('fscaled.blob'):
            raw = pickle.load(open('raw.blob', 'r'))
            fscaled = pickle.load(open('fscaled.blob', 'r'))
        else:
            armies_battles_for_regr = []
            armies_battles_for_clust = {'P': [], 'T': [], 'Z': []}
            # [[P1units],[P2units],battle_result]
            fnamelist = []
            if sys.argv[1] == '-d': # -d for directory
                import glob
                fnamelist = glob.iglob(sys.argv[2] + '/*.rgd')
            else:
                fnamelist = sys.argv[1:]
            for fname in fnamelist:
                f = open(fname)
                players_races = data_tools.players_races(f)
                armies_raw = extract_armies_battles(f)
                if SCORES_REGRESSION:
                    battles_r = map(functools.partial(format_battle_for_regr,
                            players_races), armies_raw)
                    armies_battles_for_regr.extend(battles_r)
                battles_c = filter(lambda x: len(x[0]) and len(x[1]),
                        map(functools.partial(format_battle_for_clust_adv,
                        players_races), armies_raw))
                print battles_c
                for b in battles_c:
                    for k in armies_battles_for_clust.iterkeys():
                        armies_battles_for_clust[k].extend(b[k])
                # TODO clustering advanced with only efficient battles/armies
                # TODO adversary classification (what works against what)

            if SCORES_REGRESSION:
                armies_battles_regr_raw = np.array(armies_battles_for_regr, np.float32)
                armies_battles_regr_fscaled = data_tools.features_scaling(armies_battles_regr_raw)
                from sklearn import linear_model

                import pylab as pl
                from sklearn.decomposition import PCA, FastICA
                from sklearn import cross_validation
                from sklearn import metrics

                X = armies_battles_regr_fscaled[:,:-1]
                Y = armies_battles_regr_fscaled[:,-1]
                clf = linear_model.LinearRegression()
                print clf.fit(X, Y)
                print clf.score(X, Y)
                scores = cross_validation.cross_val_score(
                             clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
                print scores

                pca = PCA(1)
                X_r = pca.fit(X).transform(X)
                print clf.fit(X_r, Y)
                print clf.score(X_r, Y)
                scores = cross_validation.cross_val_score(
                             clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
                print scores
                #pl.scatter(X_r, Y, color='black')
                #pl.plot(X_r, clf.predict(X_r), color='blue', linewidth=3)
                #pl.xticks(())
                #pl.yticks(())
                #pl.show()

                pca = PCA(2)
                X_r = pca.fit(X).transform(X)
                print clf.fit(X_r, Y)
                print clf.score(X_r, Y)
                scores = cross_validation.cross_val_score(
                             clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
                print scores
                #pl.scatter(X_r[:,0], X_r[:,1], color='black')
                #pl.plot(X_r, clf.predict(X_r), color='blue', linewidth=3)
                #pl.xticks(())
                #pl.yticks(())
                #pl.show()

                clf = linear_model.BayesianRidge()
                print clf.fit(X, Y)
                print clf.score(X, Y)
                scores = cross_validation.cross_val_score(
                             clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
                print scores

                clf = linear_model.Lasso(alpha = 0.1)
                print clf.fit(X, Y)
                print clf.score(X, Y)
                scores = cross_validation.cross_val_score(
                             clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
                print scores

                from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

                clf = LassoLarsIC(criterion = 'bic')
                print clf.fit(X, Y)
                print clf.score(X, Y)
                scores = cross_validation.cross_val_score(
                             clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
                print scores

                clf = LassoCV(cv=5)
                print clf.fit(X, Y)
                print clf.score(X, Y)
                scores = cross_validation.cross_val_score(
                             clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
                print scores

                #import ldavb
                #clust = LDAVB()
                #print clust.fit(armies_battles_for_clust['P'])
                #print clust.fit(armies_battles_for_clust['T'])
                #print clust.fit(armies_battles_for_clust['Z'])
                
