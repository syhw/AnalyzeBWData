import sys, shutil, os

NUM_WORKERS = 5

def filter_prefix(prefix):
    if not os.path.isdir(prefix+'not_1v1'):
        os.mkdir(prefix+'not_1v1')
    print 'working on', prefix
    for dname in os.listdir(prefix):
        if os.path.isdir(prefix+dname):
            print dname
            to_move = []
            for fname in os.listdir(prefix + dname):
                if ".rep.rgd" in fname:
                    fullfn = prefix + dname + '/' + fname
                    pworkers = {}
                    f = open(fullfn)
                    init = False
                    players = False
                    for line in f:
                        if not init:
                            if "The following players are in this replay:" in line:
                                players = True
                            if "Begin replay data:" in line:
                                init = True
                            if players and not init:
                                pworkers[line.split(',')[0]] = 0
                        if init:
                            if len(pworkers.keys()) < 3:
                                break
                            if "Created" in line:
                                if "Protoss Probe" in line or "Terran SCV" in line or "Zerg Drone" in line:
                                    pworkers[line.split(',')[1]] += 1
                                    np = 0
                                    for nw in pworkers.itervalues():
                                        if nw > NUM_WORKERS:
                                            np += 1
                                    if np > 2:
                                        to_move.append(fullfn[:-3])
                                        break

            for fn in to_move:
                print "should move", fn
                #shutil.move(fn+".rld", prefix+'not_1v1/'+fn+".rld")
                #shutil.move(fn+".rgd", prefix+'not_1v1/'+fn+".rgd")
                #shutil.move(fn+".rod", prefix+'not_1v1/'+fn+".rod")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filter_prefix(sys.argv[1])
    else:
        filter_prefix('replays/')

