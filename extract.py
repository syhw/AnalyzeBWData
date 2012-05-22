import glob, sys, re, collections

template = ' & & & '
d = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: 0)))

map_mu = {'PvP':0, 'PvT':2, 'PvZ': 4, 'TvT': 6, 'TvZ': 8, 'ZvZ':10}

for fn in glob.iglob(sys.argv[1]+'/*.out'):
    print fn
    f = open(fn)
    name = fn.split('/')[1]
    mu = name[:3]
    full = 0
    if 'w' in name or 's' in name:
        full = 1
    for line in f:
        if 'simple outcome predictor performance:' in line:
            s = 0
            try:
                s = int(round(100*float(re.findall('\d\.\d ', line)[0])))
            except:
                pass
            if s == 0:
                try:
                    s = int(round(100*float(re.findall('\d\.\d\d ', line)[0])))
                except:
                    pass
            if s == 0:
                s = int(round(100*float(re.findall('\d\.\d\d\d', line)[0])))
            if s > d[mu][full][0]:
                d[mu][full][0] = s
        elif 'cluster only outcome predictor perfor' in line:
            s = 0
            try:
                s = int(round(100*float(re.findall('\d\.\d ', line)[0])))
            except:
                pass
            if s == 0:
                try:
                    s = int(round(100*float(re.findall('\d\.\d\d ', line)[0])))
                except:
                    pass
            if s == 0:
                s = int(round(100*float(re.findall('\d\.\d\d\d', line)[0])))
            if s > d[mu][full][1]:
                d[mu][full][1] = s
        elif 'most prob cluster only outcome predic' in line:
            s = 0
            try:
                s = int(round(100*float(re.findall('\d\.\d ', line)[0])))
            except:
                pass
            if s == 0:
                try:
                    s = int(round(100*float(re.findall('\d\.\d\d ', line)[0])))
                except:
                    pass
            if s == 0:
                s = int(round(100*float(re.findall('\d\.\d\d\d', line)[0])))
            if s > d[mu][full][1]:
                d[mu][full][1] = s
        elif '(score * cluster factor) outcome pred' in line:
            s = 0
            try:
                s = int(round(100*float(re.findall('\d\.\d ', line)[0])))
            except:
                pass
            if s == 0:
                try:
                    s = int(round(100*float(re.findall('\d\.\d\d ', line)[0])))
                except:
                    pass
            if s == 0:
                s = int(round(100*float(re.findall('\d\.\d\d\d', line)[0])))
            if s > d[mu][full][2]:
                d[mu][full][2] = s
        elif '(score * most prob cluster) outcome p' in line:
            s = 0
            try:
                s = int(round(100*float(re.findall('\d\.\d ', line)[0])))
            except:
                pass
            if s == 0:
                try:
                    s = int(round(100*float(re.findall('\d\.\d\d ', line)[0])))
                except:
                    pass
            if s == 0:
                s = int(round(100*float(re.findall('\d\.\d\d\d', line)[0])))
            if s > d[mu][full][2]:
                d[mu][full][2] = s

tmp = [['00' for i in range(12)] for j in range(3)]

for mu, v in d.iteritems():
    for column, vv in v.iteritems():
        for score, vvv in vv.iteritems():
            tmp[score][map_mu[mu]+column] = str(vvv)

for line in tmp:
    print template + ' & '.join(line)


