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

    def detect_observers(line):
        if len(obs) == 0 and 'Created' in line:
            l = line.split(',')
            if l[1] not in num_created:
                num_created[l[1]] = 1
            else:
                num_created[l[1]] += 1
            if int(l[0]) > 4320: # 3 minutes * 60 seconds * 24 frames/s
                for k,v in num_created.iteritems():
                    if v < 8: # 5 workers + 1 townhall = 6 created by starting
                        obs.append(k)

    def heuristics_remove_observers(d):
        """ look if there are more than 2 players engaged in the battle
        and seek player with nothing else than SCV and Command Centers engaged
        in the battle
        """
        if len(d[0]) > 2: # d[0] are all units involved, if len(d[0] > 2
            # it means that there are more than 2 players in the battle
            if len(d[0]) - len(obs) > 2: # a player is not captured by the obs
                # heuristic (building SCV at the beginning)
                keys_to_del = set()
                for k,v in d[0].iteritems():
                    to_del = True
                    for unit in v:
                        if unit != 'Terran SCV' and unit != 'Terran Command Center':
                            to_del = False
                            break
                    if to_del:
                        keys_to_del.add(k)
                for kk in range(len(d)):
                    for k in keys_to_del:
                        d[kk].pop(k)
            else: # remove the observing players from the battle
                for kk in range(len(d)):
                    for k in obs:
                        d[kk].pop(k)
        return d
    
    attacks = []
    obs = []
    num_created = {}
    for line in f:
        detect_observers(line)
        if 'IsAttacked' in line:
            tmp = data_tools.parse_dicts(line, lambda x: int(x))
            # sometimes the observers are detected in the fight (their SCVs)
            tmp = heuristics_remove_observers(tmp)
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
    scores = evaluate_pop(armies_battle[2])
    wrk_scores = armies_battle[3][p1]*2 - armies_battle[3][p1]*2
    tmp.append(scores[p1] - scores[p2] + wrk_scores)
    return tmp

def format_battle_for_clust(players_races, armies_battle):
    """ take an "extract_armies_battles" formatted battle data and make it
    ready for clustering by considering only battles which were efficient
    (on a food value) per number of units and returning 2 vectors of units
    numbers per unit types """
    p1, p2 = format_battle_init(players_races, armies_battle[1])
    pop_max = evaluate_pop(armies_battle[1])
    pop_after = evaluate_pop(armies_battle[2])
    return [],[]

f = sys.stdin
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if os.path.exists('raw.blob') and os.path.exists('fscaled.blob'):
            raw = pickle.load(open('raw.blob', 'r'))
            fscaled = pickle.load(open('fscaled.blob', 'r'))
        else:
            armies_battles_for_regr = []
            armies_battles_for_clust = []
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
                battles_r = map(functools.partial(format_battle_for_regr,
                        players_races), armies_raw)
                armies_battles_for_regr.extend(battles_r)
                #battles_c = map(functools.partial(format_battle_for_clust,
                #        data_tools.players_races(f)), armies_raw)
                #armies_battles_for_clust.extend(battles_c)

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
