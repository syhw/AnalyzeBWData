class vector_X:
    """ an object created with a matchup and showing:
        - self.{terran|protoss|zerg}_buildings_enum which maps integers to
        building names, one for each of the three races
        - self.vector_X which maps X (BuildTree) integer values to sets of
        buildings (in buildings integers) that corresponds
    """
    def __init__(self, us, them):
        matchup = them + 'v' + us
        f = open('strategy/tables/vectorx/ENUM.txt')
        self.terran_buildings_enum = []
        self.protoss_buildings_enum = []
        self.zerg_buildings_enum = []
        for line in f:
            if 'Terran_' in line:
                self.terran_buildings_enum.append(line.rstrip('\n'))
            elif 'Protoss_' in line:
                self.protoss_buildings_enum.append(line.rstrip('\n'))
            elif 'Zerg_' in line:
                self.zerg_buildings_enum.append(line.rstrip('\n'))
        f.close()
        f = open('strategy/tables/vectorx/' + matchup + '.txt')
        self.vector_X = []
        for line in f:
            self.vector_X.append(set(map(int, line.rstrip(' \n').split(' '))))

        
if __name__ == "__main__":
    v = vector_X('P', 'T')
    print v.protoss_buildings_enum
    print v.terran_buildings_enum
    print v.vector_X

