drop = ['Terran Dropship',
            'Protoss Shuttle',
            'Zerg Overlord']

military = [['Terran Marine', 'Terran Ghost', 'Terran Vulture', 'Terran Goliath', 'Terran Siege Tank Tank Mode', 'Terran Wraith', 'Terran Science Vessel', 'Terran Battlecruiser', 'Terran Siege Tank Siege Mode', 'Terran Firebat', 'Terran Medic', 'Terran Valkyrie' ], 
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
            'T' : ['Terran Marine', 'Terran Ghost', 'Terran Vulture', 'Terran Goliath', 'Terran Siege Tank Tank Mode', 'Terran Wraith', 'Terran Science Vessel', 'Terran Battlecruiser', 'Terran Siege Tank Siege Mode', 'Terran Firebat', 'Terran Medic', 'Terran Valkyrie' ], 
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

unit_double_pop = {
'Terran Marine' : 2, 'Terran Ghost' : 2, 'Terran Vulture' : 4, 'Terran Goliath': 4, 'Terran Siege Tank Tank Mode' : 4, 'Terran Wraith' : 4, 'Terran Science Vessel' : 4, 'Terran Battlecruiser' : 12, 'Terran Siege Tank Siege Mode' : 4, 'Terran Firebat' : 2, 'Terran Medic' : 2, 'Terran Valkyrie' : 6,
'Protoss Observer' : 2, 'Protoss Dragoon' : 4, 'Protoss Zealot' : 4, 'Protoss Archon' : 8, 'Protoss Reaver' : 8, 'Protoss High Templar' : 4, 'Protoss Arbiter': 8, 'Protoss Carrier' : 12, 'Protoss Scout' : 6, 'Protoss Dark Archon' : 8, 'Protoss Corsair' : 4, 'Protoss Dark Templar' : 4, 'Zerg Zergling' : 1, 'Zerg Devourer' : 4, 'Zerg Guardian' : 4, 'Zerg Ultralisk' : 8, 'Zerg Queen' : 4, 'Zerg Hydralisk' : 2, 'Zerg Mutalisk' : 4, 'Zerg Scourge' : 1, 'Zerg Lurker' : 4, 'Zerg Defiler' : 4,
'Terran SCV' : 2, 
'Protoss Probe' : 2, 
'Zerg Drone' : 2
}
