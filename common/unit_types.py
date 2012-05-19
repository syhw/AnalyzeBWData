from common.common_tools import memoize

drop = ['Terran Dropship',
        'Protoss Shuttle',
        'Zerg Overlord']

military = [['Terran Marine', 'Terran Ghost', 'Terran Vulture', 'Terran Vulture Spider Mine', 'Terran Goliath', 'Terran Siege Tank Tank Mode', 'Terran Wraith', 'Terran Science Vessel', 'Terran Battlecruiser', 'Terran Siege Tank Siege Mode', 'Terran Firebat', 'Terran Medic', 'Terran Valkyrie' ], 
        ['Protoss Observer', 'Protoss Dragoon', 'Protoss Zealot', 'Protoss Archon', 'Protoss Reaver', 'Protoss High Templar', 'Protoss Arbiter', 'Protoss Carrier', 'Protoss Scout', 'Protoss Dark Archon', 'Protoss Corsair', 'Protoss Dark Templar'], 
        ['Zerg Zergling', 'Zerg Devourer', 'Zerg Guardian', 'Zerg Ultralisk', 'Zerg Queen', 'Zerg Hydralisk', 'Zerg Mutalisk', 'Zerg Scourge', 'Zerg Lurker', 'Zerg Defiler']]

buildings = [['Terran Command Center', 'Terran Comsat Station', 'Terran Nuclear Silo', 'Terran Supply Depot', 'Terran Refinery', 'Terran Barracks', 'Terran Academy', 'Terran Factory', 'Terran Starport', 'Terran Control Tower', 'Terran Science Facility', 'Terran Covert Ops', 'Terran Physics Lab', 'Terran Machine Shop', 'Terran Engineering Bay', 'Terran Armory', 'Terran Missile Turret', 'Terran Bunker'], 
        ['Protoss Nexus', 'Protoss Robotics Facility', 'Protoss Pylon', 'Protoss Assimilator', 'Protoss Observatory', 'Protoss Gateway', 'Protoss Photon Cannon', 'Protoss Citadel of Adun', 'Protoss Cybernetics Core', 'Protoss Templar Archives', 'Protoss Forge', 'Protoss Stargate', 'Protoss Fleet Beacon', 'Protoss Arbiter Tribunal', 'Protoss Robotics Support Bay', 'Protoss Shield Battery'], 
        ['Zerg Overlord', 'Zerg Infested Command Center', 'Zerg Hatchery', 'Zerg Lair', 'Zerg Hive', 'Zerg Nydus Canal', 'Zerg Hydralisk Den', 'Zerg Defiler Mound', 'Zerg Greater Spire', 'Zerg Queens Nest', 'Zerg Evolution Chamber', 'Zerg Ultralisk Cavern', 'Zerg Spire', 'Zerg Spawning Pool', 'Zerg Creep Colony', 'Zerg Spore Colony', 'Zerg Sunken Colony', 'Zerg Extractor']]

workers = ['Terran SCV', 
        'Protoss Probe', 
        'Zerg Drone']

class by_race:
    drop = {
            'T' : 'Terran Dropship',
            'P' : 'Protoss Shuttle',
            'Z' : 'Zerg Overlord'}

    military = {
            'T' : ['Terran Marine', 'Terran Ghost', 'Terran Vulture', 'Terran Vulture Spider Mine', 'Terran Goliath', 'Terran Siege Tank Tank Mode', 'Terran Wraith', 'Terran Science Vessel', 'Terran Battlecruiser', 'Terran Siege Tank Siege Mode', 'Terran Firebat', 'Terran Medic', 'Terran Valkyrie' ], 
            'P' : ['Protoss Observer', 'Protoss Dragoon', 'Protoss Zealot', 'Protoss Archon', 'Protoss Reaver', 'Protoss High Templar', 'Protoss Arbiter', 'Protoss Carrier', 'Protoss Scout', 'Protoss Dark Archon', 'Protoss Corsair', 'Protoss Dark Templar'], 
            'Z' : ['Zerg Zergling', 'Zerg Devourer', 'Zerg Guardian', 'Zerg Ultralisk', 'Zerg Queen', 'Zerg Hydralisk', 'Zerg Mutalisk', 'Zerg Scourge', 'Zerg Lurker', 'Zerg Defiler']}

    buildings = {
            'T' : ['Terran Command Center', 'Terran Comsat Station', 'Terran Nuclear Silo', 'Terran Supply Depot', 'Terran Refinery', 'Terran Barracks', 'Terran Academy', 'Terran Factory', 'Terran Starport', 'Terran Control Tower', 'Terran Science Facility', 'Terran Covert Ops', 'Terran Physics Lab', 'Terran Machine Shop', 'Terran Engineering Bay', 'Terran Armory', 'Terran Missile Turret', 'Terran Bunker'], 
            'P' : ['Protoss Nexus', 'Protoss Robotics Facility', 'Protoss Pylon', 'Protoss Assimilator', 'Protoss Observatory', 'Protoss Gateway', 'Protoss Photon Cannon', 'Protoss Citadel of Adun', 'Protoss Cybernetics Core', 'Protoss Templar Archives', 'Protoss Forge', 'Protoss Stargate', 'Protoss Fleet Beacon', 'Protoss Arbiter Tribunal', 'Protoss Robotics Support Bay', 'Protoss Shield Battery'], 
            'Z' : ['Zerg Overlord', 'Zerg Infested Command Center', 'Zerg Hatchery', 'Zerg Lair', 'Zerg Hive', 'Zerg Nydus Canal', 'Zerg Hydralisk Den', 'Zerg Defiler Mound', 'Zerg Greater Spire', 'Zerg Queens Nest', 'Zerg Evolution Chamber', 'Zerg Ultralisk Cavern', 'Zerg Spire', 'Zerg Spawning Pool', 'Zerg Creep Colony', 'Zerg Spore Colony', 'Zerg Sunken Colony', 'Zerg Extractor']}

    workers = {
            'T' : 'Terran SCV', 
            'P' : 'Protoss Probe', 
            'Z' : 'Zerg Drone'}

    static_defense = {
            'T' : ['Terran Bunker', 'Terran Missile Turret'],
            'P' : ['Protoss Shield Battery', 'Protoss Photon Cannon'],
            'Z' : ['Zerg Creep Colony', 'Zerg Spore Colony', 'Zerg Sunken Colony']}

unit_double_pop = {
'Terran Marine' : 2, 'Terran Ghost' : 2, 'Terran Vulture' : 4, 'Terran Goliath' : 4, 'Terran Siege Tank Tank Mode' : 4, 'Terran Wraith' : 4, 'Terran Science Vessel' : 4, 'Terran Battlecruiser' : 12, 'Terran Siege Tank Siege Mode' : 4, 'Terran Firebat' : 2, 'Terran Medic' : 2, 'Terran Valkyrie' : 6,
'Protoss Observer' : 2, 'Protoss Dragoon' : 4, 'Protoss Zealot' : 4, 'Protoss Archon' : 8, 'Protoss Reaver' : 8, 'Protoss High Templar' : 4, 'Protoss Arbiter' : 8, 'Protoss Carrier' : 12, 'Protoss Scout' : 6, 'Protoss Dark Archon' : 8, 'Protoss Corsair' : 4, 'Protoss Dark Templar' : 4, 'Zerg Zergling' : 1, 'Zerg Devourer' : 4, 'Zerg Guardian' : 4, 'Zerg Ultralisk' : 8, 'Zerg Queen' : 4, 'Zerg Hydralisk' : 2, 'Zerg Mutalisk' : 4, 'Zerg Scourge' : 1, 'Zerg Lurker' : 4, 'Zerg Defiler' : 4, 'Zerg Infested Terran' : 2,
'Terran Dropship' : 4,
'Protoss Shuttle' : 4,
'Zerg Overlord' : 4, # this is FALSE but counts towards the score fairly
'Terran SCV' : 2, 
'Protoss Probe' : 2, 
'Zerg Drone' : 2
}

unit_min_price = {
'Terran Marine' : 50, 'Terran Ghost' : 25, 'Terran Vulture' : 75, 'Terran Goliath' : 100, 'Terran Siege Tank Tank Mode' : 150, 'Terran Wraith' : 150, 'Terran Science Vessel' : 100, 'Terran Battlecruiser' : 400, 'Terran Siege Tank Siege Mode' : 150, 'Terran Firebat' : 50, 'Terran Medic' : 50, 'Terran Valkyrie' : 250,'Terran Missile Turret' : 75, 'Terran Bunker' : 100, 'Protoss Observer' : 25, 'Protoss Dragoon' : 125, 'Protoss Zealot' : 100, 'Protoss Archon' : 100, 'Protoss Reaver' : 200, 'Protoss High Templar' : 50, 'Protoss Arbiter' : 100, 'Protoss Carrier' : 350, # + 200 of interceptors
'Protoss Interceptor' : 25,
'Protoss Scout' : 275, 'Protoss Dark Archon' : 250, 'Protoss Corsair' : 150, 'Protoss Dark Templar' : 125, 'Zerg Zergling' : 25, 'Zerg Devourer' : 250, 'Zerg Guardian' : 150, 'Zerg Ultralisk' : 200, 'Zerg Queen' : 100, 'Zerg Hydralisk' : 75, 'Zerg Mutalisk' : 100, 'Zerg Scourge' : 13, 'Zerg Lurker' : 125, 'Zerg Defiler' : 50, 'Zerg Infested Terran' : 100,
'Terran Dropship' : 100,
'Protoss Shuttle' : 200,
'Zerg Overlord' : 100,
'Terran SCV' : 50, 
'Protoss Probe' : 50, 
'Zerg Drone' : 50
}

unit_gas_price = {
'Terran Ghost' : 2, 'Terran Goliath' : 4, 'Terran Siege Tank Tank Mode' : 4, 'Terran Wraith' : 4, 'Terran Science Vessel' : 4, 'Terran Battlecruiser' : 12, 'Terran Siege Tank Siege Mode' : 4, 'Terran Firebat' : 2, 'Terran Medic' : 2, 'Terran Valkyrie' : 6,
'Protoss Observer' : 75, 'Protoss Dragoon' : 50, 'Protoss Archon' : 300, 'Protoss Reaver' : 100, 'Protoss High Templar' : 150, 'Protoss Arbiter' : 350, 'Protoss Carrier' : 250, 'Protoss Scout' : 125, 'Protoss Dark Archon' : 200, 'Protoss Corsair' : 100, 'Protoss Dark Templar' : 100, 'Zerg Devourer' : 150, 'Zerg Guardian' : 200, 'Zerg Ultralisk' : 200, 'Zerg Queen' : 100, 'Zerg Hydralisk' : 25, 'Zerg Mutalisk' : 100, 'Zerg Scourge' : 38, 'Zerg Lurker' : 125, 'Zerg Defiler' : 150, 'Zerg Infested Terran' : 50
}

military_set = set(drop)
for l in military:
    military_set.update(l)
military_set.update(['Terran Missile Turret', 'Terran Bunker', 'Protoss Photon Cannon', 'Protoss Shield Battery', 'Zerg Sunken Colony', 'Zerg Spore Colony'])

flying_set = set(['Terran Wraith', 'Terran Science Vessel', 'Terran Battlecruiser', 'Terran Valkyrie',
        'Protoss Observer', 'Protoss Arbiter', 'Protoss Carrier', 'Protoss Scout', 'Protoss Corsair',
        'Zerg Devourer', 'Zerg Guardian', 'Zerg Queen', 'Zerg Mutalisk', 'Zerg Scourge',
        'Terran Dropship',
        'Protoss Shuttle',
        'Zerg Overlord'])

shoot_up_set = set(['Terran Marine', 'Terran Ghost', 'Terran Goliath', 'Terran Wraith', 'Terran Science Vessel', 'Terran Battlecruiser', 'Terran Medic', 'Terran Valkyrie', 'Terran Missile Turret', 'Terran Bunker', 
        'Protoss Observer', 'Protoss Dragoon', 'Protoss Archon', 'Protoss High Templar', 'Protoss Arbiter', 'Protoss Carrier', 'Protoss Scout', 'Protoss Dark Archon', 'Protoss Corsair', 'Protoss Photon Cannon', 'Protoss Shield Battery',
        'Zerg Devourer', 'Zerg Queen', 'Zerg Hydralisk', 'Zerg Mutalisk', 'Zerg Scourge', 'Zerg Defiler', 'Zerg Spore Colony'])

dont_shoot_down_set = set(['Terran Valkyrie', 'Terran Missile Turret', 'Protoss Corsair', 'Zerg Devourer', 'Zerg Scourge', 'Zerg Spore Colony'])

shoot_down_set = military_set - dont_shoot_down_set

ground_set = military_set - flying_set

detectors_set = set(['Terran Science Vessel', 'Terran Missile Turret', 'Protoss Observer', 'Protoss Photon Cannon', 'Zerg Overlord', 'Zerg Spore Colony'])

invis_attack_set = set(['Terran Wraith', 'Terran Ghost', 'Protoss Dark Templar', 'Zerg Lurker'])

fly_tech = set(['Terran Starport', 'Protoss Stargate', 'Zerg Spire'])

invis_tech = [['Terran Covert Ops', 'Terran Academy'], ['Terran Control Tower'], ['Protoss Templar Archives'], ['Zerg Lair', 'Zerg Hydralisk Den']]

drop_tech = ['Terran Control Tower', 'Protoss Robotics Facility', 'Zerg Lair']

static_defense_set = set(['Terran Bunker', 'Terran Missile Turret',
            'Protoss Shield Battery', 'Protoss Photon Cannon',
            'Zerg Creep Colony', 'Zerg Spore Colony', 'Zerg Sunken Colony'])

static_defense_shoot_down_set = static_defense_set and shoot_down_set

static_defense_shoot_up_set = static_defense_set and shoot_up_set

required_for = {
        'Terran Marine': ['Terran Barracks'], 
        'Terran Ghost': ['Terran Barracks', 'Terran Academy', 'Terran Science Facility', 'Terran Covert Ops'], 
        'Terran Vulture': ['Terran Factory'],
        'Terran Vulture Spider Mine': ['Terran Factory', 'Terran Machine Shop'],
        'Terran Goliath': ['Terran Factory', 'Terran Armory'],
        'Terran Siege Tank Tank Mode': ['Terran Factory', 'Terran Machine Shop'],
        'Terran Siege Tank Siege Mode': ['Terran Factory', 'Terran Machine Shop'],
        'Terran Wraith': ['Terran Starport'],
        'Terran Science Vessel': ['Terran Starport', 'Terran Control Tower', 'Terran Science Facility'],
        'Terran Battlecruiser': ['Terran Starport', 'Terran Control Tower', 'Terran Science Facility', 'Terran Physics Lab'],
        'Terran Firebat': ['Terran Barracks', 'Terran Academy'],
        'Terran Medic': ['Terran Barracks', 'Terran Academy'],
        'Terran Valkyrie': ['Terran Starport', 'Terran Control Tower', 'Terran Armory'],
        'Protoss Observer': ['Protoss Robotics Facility', 'Protoss Observatory'],
        'Protoss Dragoon': ['Protoss Gateway', 'Protoss Cybernetics Core'],
        'Protoss Zealot': ['Protoss Gateway'],
        'Protoss Archon': ['Protoss Gateway', 'Protoss Templar Archives'],
        'Protoss Reaver': ['Protoss Robotics Facility', 'Protoss Robotics Bay'],
        'Protoss High Templar': ['Protoss Gateway', 'Protoss Templar Archives'],
        'Protoss Arbiter': ['Protoss Stargate', 'Protoss Arbiter Tribunal'],
        'Protoss Carrier': ['Protoss Stargate', 'Protoss Fleet Beacon'],
        'Protoss Scout': ['Protoss Stargate'],
        'Protoss Dark Archon': ['Protoss Gateway', 'Protoss Templar Archives'],
        'Protoss Corsair': ['Protoss Stargate'],
        'Protoss Dark Templar': ['Protoss Gateway', 'Protoss Templar Archives'],
        'Zerg Zergling': ['Zergling Spawning Pool'],
        'Zerg Devourer': ['Zerg Greater Spire'],
        'Zerg Guardian': ['Zerg Greater Spire'],
        'Zerg Ultralisk': ['Zerg Ultralisk Cavern'],
        'Zerg Queen': ['Zerg Queens Nest'],
        'Zerg Hydralisk': ['Zerg Hydralisk Den'],
        'Zerg Mutalisk': ['Zerg Spire'],
        'Zerg Scourge': ['Zerg Spire'],
        'Zerg Lurker': ['Zerg Hydralisk Den', 'Zerg Lair'],
        'Zerg Defiler': ['Zerg Defiler Mound'],
        'Terran Dropship': ['Terran Starport', 'Terran Control Tower'],
        'Protoss Shuttle': ['Protoss Robotics Facility'],
        'Zerg Overlord': ['Zerg Lair']
        }


@memoize
def score_unit(unit):
    return unit_min_price.get(unit, 0) + (4.0/3)*unit_gas_price.get(unit, 0) + 25*unit_double_pop.get(unit, 0)


