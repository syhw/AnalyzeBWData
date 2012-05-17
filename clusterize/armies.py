import sys, os, pickle, copy, itertools, functools, math
from collections import defaultdict
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
WITH_STATIC_DEFENSE = False # tells if we include static defense in armies
CSV_ARMIES_OUTPUT = True

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

class ArmyCompositions:
    by_race = {}
    for race in ['T', 'P', 'Z']:
        by_race[race] = unit_types.by_race.military[race]+[unit_types.by_race.drop[race]]
        if WITH_STATIC_DEFENSE:
            by_race[race].extend(unit_types.by_race.static_defense[race])

    def __init__(self, race):
        self.race = race
        if race == 'P':
            self.basic_units = {
                    'zealot': {'Protoss Zealot': 0.4},
                    'goon': {'Protoss Dragoon': 0.3},
                    'DT': {'Protoss Dark Templar': 0.6},
                    'reaver': {'Protoss Reaver': 0.1},
                    'carrier': {'Protoss Carrier': 0.2},
                    'corsair': {'Protoss Corsair': 0.2},
                    'scout': {'Protoss Scout': 0.4},
                    'archon': {'Protoss Archon': 0.2}}
            self.special_units = {
                    'observer': {'Protoss Observer': 0.01},
                    'arbiter': {'Protoss Arbiter': 0.05},
                    'HT': {'Protoss High Templar': 0.05},
                    'darchon': {'Protoss Dark Archon': 0.05},
                    'shuttle': {'Protoss Shuttle': 0.2} # full with zealots
                    }
        elif race == 'T':
            self.basic_units = {
                    'marine': {'Terran Marine': 0.5},
                    'medic': {'Terran Medic': 0.1},
                    'firebat': {'Terran Firebat': 0.1},
                    'vulture': {'Terran Vulture': 0.3},
                    'goliath': {'Terran Goliath': 0.2},
                    'wraith': {'Terran Wraith': 0.5},
                    'battlecruiser': {'Terran Battlecruiser': 0.2},
                    'valkyrie': {'Terran Valkyrie': 0.2}}
            self.special_units = {
                    'vessel': {'Terran Science Vessel': 0.05},
                    'tank': {'Terran Siege Tank Tank Mode': 0.2}, # trick
                    'ghost': {'Terran Ghost': 0.1},
                    'dropship': {'Terran Dropship': 0.1} # full with marines
                    }
        elif race == 'Z':
            self.basic_units = {
                    'zergling': {'Zerg Zergling': 0.7},
                    'hydra': {'Zerg Hydralisk': 0.4},
                    'ultra': {'Zerg Ultralisk': 0.15},
                    'muta': {'Zerg Mutalisk': 0.5},
                    'guardian': {'Zerg Guardian': 0.2},
                    'devourer': {'Zerg Devourer': 0.2},
                    'scourge': {'Zerg Scourge': 0.2}}
            self.special_units = {                    
                    'queen': {'Zerg Queen': 0.05},
                    'lurker': {'Zerg Lurker': 0.2},
                    'defiler': {'Zerg Defiler': 0.05},
                    'overlord': {'Zerg Overlord': 0.05} # full with zerglings or detector
                    }
        self.compositions = {}
        self.compositions.update(self.basic_units)
        self.compositions.update(self.special_units)
        for unit,value in self.basic_units.iteritems():
            for sunit,svalue in self.special_units.iteritems():
                self.compositions[unit+'_'+sunit] = {}
                self.compositions[unit+'_'+sunit].update(value)
                self.compositions[unit+'_'+sunit].update(svalue)
        width = 0.01
        epsilon = 1.0e-16
        self.P_unit_knowing_cluster = {}
        for name,compo in self.compositions.iteritems():
            for unit,min_percentage in compo.iteritems():
                max_percentage = 1.0-sum([v for k,v in compo.iteritems() if k != unit])
                if unit not in self.P_unit_knowing_cluster:
                    self.P_unit_knowing_cluster[unit] = defaultdict(lambda: lambda x: width)
                def f(x):
                    if min_percentage < x < max_percentage:
                        return width*1.0/(max_percentage-min_percentage) - (1.0-max_percentage + min_percentage)*epsilon
                    else:
                        return epsilon
                self.P_unit_knowing_cluster[unit].update({name: f})

    def distrib(self, percents_list):
        d = {}
        for cluster in self.compositions:
            d[cluster] = 0.0
            for i, unit_type in enumerate(ArmyCompositions.by_race[self.race]):
                d[cluster] += math.log(self.P_unit_knowing_cluster[unit_type][cluster](percents_list[i]))
        return d


class percent_list(list):
    def new_battle(self, d):
        race = d.iterkeys().next()[0] # first character of the first unit
        tmp = [d.get(u, 0.0) for u in ArmyCompositions.by_race[race]]
        if race == 'T':
            tmp[unit_types.by_race.military[race].index('Terran Siege Tank Tank Mode')] += tmp.pop(unit_types.by_race.military[race].index('Terran Siege Tank Siege Mode'))
        self.append(tmp)


f = sys.stdin
if __name__ == "__main__":
    if len(sys.argv) > 1:
        armies_battles_for_regr = []
        armies_battles_for_clust = {'P': percent_list(), 
                'T': percent_list(), 
                'Z': percent_list()}
        units_ratio_and_scores = []
        fnamelist = []
        armies_compositions = {'P': ArmyCompositions('P'),
                'T': ArmyCompositions('T'),
                'Z': ArmyCompositions('Z')}

        if sys.argv[1] == '-d': # -d for directory
            import glob
            fnamelist = glob.iglob(sys.argv[2] + '/*.rgd')
        else:
            fnamelist = sys.argv[1:]

        for fname in fnamelist:
            f = open(fname)
            players_races = data_tools.players_races(f)

            ### Parse battles and extract armies (before, after)
            armies_raw = extract_armies_battles(f)

            if SCORES_REGRESSION:
                ### Format battles for predict/regression (of the outcome)
                battles_r = map(functools.partial(format_battle_for_regr,
                        players_races), armies_raw)
                armies_battles_for_regr.extend(battles_r)

            ### Format battles for clustering (with armies order P>T>Z)
            ### (army_p1, army_p2, final_score_p1, final_score_p2)
            battles_c = filter(lambda x: len(x[0]) and len(x[1]),
                    map(functools.partial(format_battle_for_clust_adv,
                    players_races), armies_raw))
            #print battles_c

            ### save these battles for further use
            units_ratio_and_scores.extend(battles_c)

            ### Sort armies by race and put inside battles_for_clust
            for b in battles_c:
                for race in armies_battles_for_clust.iterkeys():
                    first_unit = b[0].iterkeys().next()
                    if first_unit[0] == race:
                        armies_battles_for_clust[race].new_battle(b[0])
                    first_unit = b[1].iterkeys().next()
                    if first_unit[0] == race:
                        armies_battles_for_clust[race].new_battle(b[1])
            #print armies_battles_for_clust

        from common.parallel_coordinates import parallel_coordinates
        for race, l in armies_battles_for_clust.iteritems():
            if len(l) > 0:
                csv = open(race+'_armies.csv', 'w')
                x_l = [u for u in ArmyCompositions.by_race[race]]
                if race == 'T':
                    x_l.pop(unit_types.by_race.military[race].index('Terran Siege Tank Siege Mode'))
                x_l.append(unit_types.by_race.drop[race])
                x_l = map(lambda s: ''.join(s.split(' ')[1:]), x_l)
                #print x_l
                #parallel_coordinates(l, x_labels=x_l).savefig("parallel_"+race+".png")
                if CSV_ARMIES_OUTPUT:
                    csv.write(','.join(x_l) + '\n')
                    for line in l:
                        csv.write(','.join(map(lambda e: str(e), line))+'\n')
                for p_l in l:
                    print x_l
                    print p_l
                    print sorted([(c,logprob) for c,logprob in armies_compositions[race].distrib(p_l).iteritems()], key=lambda x: x[1], reverse=True)[:5]
                    


        from sklearn import decomposition
        from sklearn import mixture
        from sklearn import manifold
        from sklearn import cluster
        for race, list_of_percentages in armies_battles_for_clust.iteritems():
            print race
            compo = np.array(list_of_percentages)
            if len(compo):
                pca = decomposition.PCA()
                pca.fit(compo)
                print pca
                print pca.explained_variance_

                gmm = mixture.GMM(n_components=8, min_covar=0.000001, cvtype='full')
                gmm.fit(compo)
                print gmm
                print unit_types.by_race.military[race],
                print unit_types.by_race.drop[race]
                if WITH_STATIC_DEFENSE:
                    print unit_types.by_race.static_defense[race]
                print gmm.means

                dpgmm = mixture.DPGMM(cvtype='full')
                dpgmm.fit(compo)
                print dpgmm
                # print dpgmm.means
                # print dpgmm.precisions

                man = manifold.Isomap()
                man.fit(compo)
                print man
                print man.embedding_

                db = cluster.DBSCAN(eps=0.1)
                dbscan = db.fit(compo)
                print dbscan
                print dbscan.components_


            else:
                print "No battles"


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
            
