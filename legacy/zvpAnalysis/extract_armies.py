import sys, os, pickle, copy, itertools
try:
    import numpy as np
except:
    print "you need numpy"
    sys.exit(-1)

ut = {'T' : set(), 'P' : set(), 'Z' : set()}
armyut = {'T' : [], 'P' : ['Protoss Observer', 'Protoss Dragoon', 'Protoss Zealot', 'Protoss Archon', 'Protoss Reaver', 'Protoss High Templar', 'Protoss Arbiter', 'Protoss Carrier', 'Protoss Shuttle', 'Protoss Scout', 'Protoss Dark Archon', 'Protoss Corsair', 'Protoss Dark Templar'], 'Z' : ['Zerg Zergling', 'Zerg Devourer', 'Zerg Guardian', 'Zerg Ultralisk', 'Zerg Queen', 'Zerg Hydralisk', 'Zerg Mutalisk', 'Zerg Scourge', 'Zerg Lurker', 'Zerg Defiler']}
armies = {'5' : {'T' : [], 'P' : [], 'Z' : []}, '10' : {'T' : [], 'P' : [], 'Z' : []}, '15' : {'T' : [], 'P' : [], 'Z' : []}, '20' : {'T' : [], 'P' : [], 'Z' : []}}

apply_ica = False

def features_scaling(tt):
    # find max and compute mean
    mx = tt.max(axis=0)
    # because of the discrete sampling (5, 10, 15, 20 minutes), 
    # some features'values are always 0
    for i, e in enumerate(mx):
        if e == 0.0:
            mx[i] = 1.0
    me = tt.mean(axis=0)
    # divide each feature by its max
    return (tt - me) / mx

def extract_from(f):
    """ Will extract all the unit numbers from all players in replay 'f'
    and dump the unit numbers for each types at 5, 10, 15 and 20 minutes in
    the armies datastruct (through the dump function)"""
    armies_players = [] # list of tuples ('player', {'unit type' : number})
    preproc = True # true while preprocessing the header of a rep
    finished = False # true when finished processing a rep
    players_list = False # true when listing the players in the rep's header
    min5 = False # true when in game time > 5 min
    min10 = False # true when in game time > 10 min
    min15 = False # true when in game time > 15 min
    min20 = False # true when in game time > 20 min

    def dump(l, t):
        """ list l of army{'unit type' : number} / player 
            time t (minutes) """
        for ap in l:
            tmplist = []
            for elem in armyut[ap[0][0]]:
                if ap[1].has_key(elem):
                    tmplist.append(float(ap[1][elem]))
                else:
                    tmplist.append(0.0)
            armies[t][ap[0][0]].append(tmplist)

    for line in f:
        if preproc:
            if players_list:
                l = line.split(',')
                if len(l) > 1:
                    armies_players.append((l[2].strip(' '), {}))
                elif 'Begin replay data' in line:
                    preproc = False
            if 'The following players are in this replay' in line:
                players_list = True
        elif not finished:
            if 'EndGame' in line:
                finished = True
                break
            l = line.split(',')
            p = int(l[1])
            if p < 0: # removes neutrals
                continue
            if 'Created' in l[2] or 'Morph' in l[2]:
                uname = l[4].strip(' ')
                ut[uname[0]].add(uname)
                armies_players[p][1][uname] = armies_players[p][1].get(uname, 0) + 1
            elif 'Destroyed' in l[2]:
                uname = l[4].strip(' ')
                armies_players[p][1][uname] = armies_players[p][1].get(uname, 0) - 1
            elif 'PlayerLeftGame' in l[2]:
                finished = True
            if not min5 and (7200 - int(l[0])) < 0:
                min5 = True
                dump(armies_players, '5')
            elif not min10 and (14400 - int(l[0])) < 0:
                min10 = True
                dump(armies_players, '10')
            elif not min15 and (21600 - int(l[0])) < 0:
                min15 = True
                dump(armies_players, '15')
            elif not min20 and (28800 - int(l[0])) < 0:
                min20 = True
                dump(armies_players, '20')

def clusterize_dirichlet(*args, **kwargs):
    # TODO plotting when the classifier is NOT learned in the PCA(2-best) space (ellipses are wrong)
    """ Clustering and plotting with Dirichlet process GMM """
    ### Clustering
    try:
        from sklearn import mixture
        from scipy import linalg
        import pylab as pl
        import matplotlib as mpl
        from sklearn.decomposition import PCA, FastICA
    except:
        print "You need SciPy and scikit-learn"
        sys.exit(-1)

    models = []
    for arg in args:
        if apply_ica:
            for featurenb in range(len(arg[0])):
                if sum(arg[:, featurenb]) == 0.0:
                    arg = arg[:, range(featurenb+1)+range(featurenb+1, len(arg[0]))]
        if kwargs.get('em_gmm', False):
            dpgmm = mixture.GMM(n_components = 4, cvtype='full')
        else:
            dpgmm = mixture.DPGMM(n_components = 100, cvtype='full', alpha=1000.0)
        if kwargs.get('clf_on_pca', False):
            pca = PCA(2)
            dpgmm.fit(pca.fit(arg).transform(arg))
        else:
            dpgmm.fit(arg)
        print dpgmm
        models.append(copy.deepcopy(dpgmm))
        print raw_input("press any key to pass")

    ### Plotting
    color_iter = itertools.cycle (['r', 'g', 'b', 'c', 'm'])
    for i, (clf, data) in enumerate(zip(models, args)):
        if apply_ica:
            ica = FastICA(2)
            X_r = ica.fit(data).transform(data)
            print ica.get_mixing_matrix()
        else:
            pca = PCA(2)
            X_r = pca.fit(data).transform(data)
        print data
        print X_r
        print raw_input("press any key to pass")
        splot = pl.subplot((len(args)+1)/ 2, 2, 1+i)
        pl.scatter(X_r[:,0], X_r[:,1])
        if kwargs.get('clf_on_pca', False):
            Y_ = clf.predict(X_r)
        else:
            Y_ = clf.predict(data)
        for i, (mean, covar, color) in enumerate(zip(clf.means, clf.covars,
                                                     color_iter)):
            v, w = linalg.eigh(covar)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            #pl.scatter(data[Y_== i, 0], data[Y_== i, 1], .8, color=color)
            pl.scatter(X_r[Y_== i, 0], X_r[Y_== i, 1], .8, color=color)
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1]/u[0])
            angle = 180 * angle / np.pi # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
        pl.xlim(X_r[:, 0].min(), X_r[:, 0].max())
        pl.ylim(X_r[:, 1].min(), X_r[:, 1].max())
        pl.xticks(())
        pl.yticks(())
        pl.title("Dirichlet process GMM")
    pl.show()

def clusterize_r_em(*args, **kwargs):
    """ Clustering and plotting with EM GMM"""
    try:
        from rpy2.robjects import r
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        from sklearn.decomposition import PCA
    except:
        print "You need rpy2"
        sys.exit(-1)

    r.library("mclust")
    for arg in args:
        if kwargs.get('clf_on_pca', False):
            pca = PCA(2)
            arg = pca.fit(arg).transform(arg)
        model = r.Mclust(arg)
        print model
        print r.summary(model)
        r.quartz("plot")
        r.plot(model, arg)
        print raw_input("press any key to pass")

def dump_csv(a, fn, aut):
    f = open(fn, 'w')
    for i, ut in enumerate(aut):
        f.write(ut)
        if i == len(aut) - 1:
            f.write('\n')
        else:
            f.write(',')
    for line in a:
        for i, e in enumerate(line):
            f.write(str(e))
            if i == len(line) - 1:
                f.write('\n')
            else:
                f.write(',')

#TEST print features_scaling(np.array([[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0]]))

f = sys.stdin
if len(sys.argv) > 1:
    if os.path.exists('fscaled1.blob') and os.path.exists('fscaled2.blob') and os.path.exists('fscaled3.blob') and os.path.exists('fscaled4.blob'):
        fscaled1 = pickle.load(open('fscaled1.blob', 'r'))
        fscaled2 = pickle.load(open('fscaled2.blob', 'r'))
        fscaled3 = pickle.load(open('fscaled3.blob', 'r'))
        fscaled4 = pickle.load(open('fscaled4.blob', 'r'))
    else:
        if sys.argv[1] == '-d':
            import glob
            for fname in glob.iglob(sys.argv[2] + '/*.rgd'):
                f = open(fname)
                extract_from(f)
        else:
            for arg in sys.argv[1:]:
                f = open(arg)
                extract_from(f)
        print ut
        armies_np5 = {'T' : np.array(armies['5']['T']), 'P' : np.array(armies['5']['P']), 'Z' : np.array(armies['5']['Z'])}
        armies_np10 = {'T' : np.array(armies['10']['T']), 'P' : np.array(armies['10']['P']), 'Z' : np.array(armies['10']['Z'])}
        armies_np15 = {'T' : np.array(armies['15']['T']), 'P' : np.array(armies['15']['P']), 'Z' : np.array(armies['15']['Z'])}
        armies_np20 = {'T' : np.array(armies['20']['T']), 'P' : np.array(armies['20']['P']), 'Z' : np.array(armies['20']['Z'])}
        fscaled1 = features_scaling(armies_np5['P'])
        fscaled1.dump('fscaled1.blob')
        fscaled2 = features_scaling(armies_np10['P'])
        fscaled2.dump('fscaled2.blob')
        fscaled3 = features_scaling(armies_np15['P'])
        fscaled3.dump('fscaled3.blob')
        fscaled4 = features_scaling(armies_np20['P'])
        fscaled4.dump('fscaled4.blob')

    if sys.argv[1] == '-e':
        dump_csv(fscaled1, 'fscaled1.csv', armyut['P'])
        dump_csv(fscaled2, 'fscaled2.csv', armyut['P'])
        dump_csv(fscaled3, 'fscaled3.csv', armyut['P'])
        dump_csv(fscaled4, 'fscaled4.csv', armyut['P'])
    else:
        clusterize_dirichlet(fscaled1, fscaled2, fscaled3, fscaled4, clf_on_pca=True)
        clusterize_r_em(fscaled1, fscaled2, fscaled3, fscaled4, clf_on_pca=True)
