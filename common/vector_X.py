from common.common_tools import memoize

class vector_X:
    """ 
    an object created with a matchup and showing:
        - self.enums which maps integers to
        building names, one for each of the three races
        - self.vector_X which maps X (BuildTree) integer values to sets of
        buildings (in buildings integers) for race 'them'
        - self.enum which maps buildings names to their int value in sets
        for the race of 'them' (vector_X at hand)
    """
    def __init__(self, us, them):
        matchup = them + 'v' + us
        f = open('strategy/tables/vectorx/ENUM.txt')
        self.enums = {'P': [], 'T': [], 'Z': []}
        for line in f:
            if 'Terran_' in line:
                self.enums['T'].append(line.rstrip('\n'))
            elif 'Protoss_' in line:
                self.enums['P'].append(line.rstrip('\n'))
            elif 'Zerg_' in line:
                self.enums['Z'].append(line.rstrip('\n'))
        f.close()
        self.enum = self.enums[them]
        f = open('strategy/tables/vectorx/' + matchup + '.txt')
        self.vector_X = []
        for line in f:
            self.vector_X.append(set(map(int, line.rstrip(' \n').split(' '))))


    @memoize
    def index_enum(self, building):
        return self.enum.index(building)

        
if __name__ == "__main__":
    v = vector_X('P', 'T')
    print v.enums['P']
    print v.enums['T']
    print v.vector_X
    print len(v.vector_X)

