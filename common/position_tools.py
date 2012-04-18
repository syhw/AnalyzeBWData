from scipy.spatial import cKDTree

class hashable_pos_list(list):
    def __hash__(self):
        return (self[0], self[1]).__hash__()

class PositionMapper:
    """ 
    Finds the CDR and Region of the attack prior to 
    https://github.com/SnippyHolloW/bwrepdump/commit/aa18efd053b9d83f7c4de5c879c59e0db8fb7534
    which dumps it now! (in new datasets, how stupid was I...)

    Uses two kd-trees and all the positions of the units as sampling of the map
    """
    def __init__(self, f):
        """ f should be a rld (replay location data) opened file """
        self.pos_to_CDR = {}
        self.pos_to_Reg = {}
        started = False
        posline = ''
        to_add_kd_CDR = set()
        to_add_kd_Reg = set()
        for line in f:
            line = line.rstrip('\r\n')
            if '[Replay Start]' in line:
                started = True
            if started:
                if 'CDR' in line:
                    p = posline.split(',')
                    p = hashable_pos_list([int(p[2]),int(p[3])])
                    self.pos_to_CDR[p] = int(line.split(',')[3])
                    to_add_kd_CDR.add(p)
                elif 'Reg' in line:
                    p = posline.split(',')
                    p = hashable_pos_list([int(p[2]),int(p[3])])
                    self.pos_to_Reg[p] = int(line.split(',')[3])
                    to_add_kd_Reg.add(p)
                else:
                    posline = line
        self.kd_CDR = cKDTree(list(to_add_kd_CDR))
        self.kd_Reg = cKDTree(list(to_add_kd_Reg))
    def get_CDR(self, x, y):
        return self.pos_to_CDR[hashable_pos_list(self.kd_CDR.data[self.kd_CDR.query([x, y])[1]])]
    def get_Reg(self, x, y):
        return self.pos_to_Reg[hashable_pos_list(self.kd_Reg.data[self.kd_Reg.query([x, y])[1]])]

class DistancesMaps:
    def __init__(self, f):
        """ f should be a rld (replay location data) opened file """
        self.dist_Reg = {} # dist_Reg[r1][r2] = ground pathfinding distance
        self.dist_CDR = {}
        self.max_dist = 0.0
        i_l = []
        Reg = False
        CDR = False
        for line in f:
            if '[Replay Start]' in line:
                break
            if 'Regions' in line:
                i_l = [int(x) for x in line.split(',')[1:]]
                Reg = True
                CDR = False
                continue
            elif 'ChokeDepReg' in line:
                i_l = [int(x) for x in line.split(',')[1:]]
                Reg = False
                CDR = True
                continue
            if Reg:
                l = line.split(',')
                d_l = [float(x) for x in l[1:]]
                j = int(l[0])
                for i, e in enumerate(d_l):
                    if e > self.max_dist:
                        self.max_dist = e
                    if j in self.dist_Reg:
                        self.dist_Reg[j][i_l[i]] = e
                    else:
                        self.dist_Reg[j] = {i_l[i] : e}
                    if i_l[i] in self.dist_Reg:
                        self.dist_Reg[i_l[i]][j] = e
                    else:
                        self.dist_Reg[i_l[i]] = {j : e}
            elif CDR:
                l = line.split(',')
                d_l = [float(x) for x in l[1:]]
                j = int(l[0])
                for i, e in enumerate(d_l):
                    if e > self.max_dist:
                        self.max_dist = e
                    if j in self.dist_CDR:
                        self.dist_CDR[j][i_l[i]] = e
                    else:
                        self.dist_CDR[j] = {i_l[i] : e}
                    if i_l[i] in self.dist_CDR:
                        self.dist_CDR[i_l[i]][j] = e
                    else:
                        self.dist_CDR[i_l[i]] = {j : e}
    def dist(self, r1, r2, t='Reg'):
        if r1 < 0 or r2 < 0: # when one of the player no longer has any base
            return self.max_dist # but state.players_bases[p] returns -1
            # TODO check
        elif r1 == r2:
            return 0.0
        elif t == 'Reg':
            return self.dist_Reg[r1][r2] 
        elif t == 'CDR':
            return self.dist_CDR[r1][r2]
        else:
            print "TYPE ERROR"
            return "TYPE ERROR"
    def list_regions(self, t='Reg'):
        if t == 'Reg':
            return self.dist_Reg.iterkeys()
        else:
            return self.dist_CDR.iterkeys()

    
