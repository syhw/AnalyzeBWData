#include "tech_estimator.h"	
#include "enums_name_tables_tt.h"
#include <fstream>

#define LEARNED_TIME_LIMIT 1080 // 18 minutes
#define MIN_PROB 0.000000000000000001

using namespace std;

void tech_estimator::loadTable(const char* tname)
{
    std::ifstream ifs(tname);
    boost::archive::text_iarchive ia(ifs);
    ia >> st;
}

tech_estimator::tech_estimator(const string& matchup)
: notFirstOverlord(false)
, hasInfered(false)
{
/// Load the learned prob tables (uniforms+bell shapes) for the right match up
/// The learning of these tables is described in the CIG 2011 Paper
/// A Bayesian Model for Opening Prediction in RTS Games with Application to StarCraft, Gabriel Synnaeve, Pierre Bessière, CIG (IEEE) 2011
/// Code for the learning is here: https://github.com/SnippyHolloW/OpeningTech
    {
        string serializedTablesFileName("tables/gauss/");
        serializedTablesFileName.append(matchup);
        serializedTablesFileName.append(".table");
        loadTable(serializedTablesFileName.c_str());
    }

    /// Initialize openingsProbas with a uniform distribution
    size_t nbOpenings = st.openings.size();
    for (size_t i = 0; i < nbOpenings; ++i)
        openingsProbas.push_back(1.0 / nbOpenings); 
}

tech_estimator::~tech_estimator()
{
}

void tech_estimator::onUnitShow(const string& unit_name, int frame)
{
    /// We only get shown enemy buildings + overlords
    /// We are interested in the time at which the construction began
    /// The called infered the buildings needed to produce viewed units
    /// and the time (frame parameter) at which the building was built
    if (frame/24 >= LEARNED_TIME_LIMIT)
        return; // TODO
    insertBuilding(unit_name);
    computeDistribOpenings(frame);
}
/// st.openings[i].c_str(), openingsProbas[i]*100

bool tech_estimator::alreadySaw(const string& ut)
{
    bool ret = (_alreadySawUnitTypes.find(ut) != _alreadySawUnitTypes.end());
    if (!ret)
        _alreadySawUnitTypes.insert(ut);
    return ret;
}

bool tech_estimator::insertBuilding(const string& ut)
{
    size_t previous_size = buildingsTypesSeen.size();
    
    /// Protoss
    if (ut.compare("Protoss Nexus"))
    {
        int tmp = Protoss_Nexus;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Protoss_Nexus3)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Protoss_Robotics_Facility"))
        buildingsTypesSeen.insert(Protoss_Robotics_Facility);
    else if (ut.compare("Protoss_Pylon"))
    {
        int tmp = Protoss_Pylon;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Protoss_Pylon3)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Protoss_Assimilator"))
    {
        int tmp = Protoss_Assimilator;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Protoss_Assimilator2)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Protoss_Observatory"))
        buildingsTypesSeen.insert(Protoss_Observatory);
    else if (ut.compare("Protoss_Gateway"))
    {
        int tmp = Protoss_Gateway;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Protoss_Gateway4)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Protoss_Photon_Cannon"))
        buildingsTypesSeen.insert(Protoss_Photon_Cannon);
    else if (ut.compare("Protoss_Citadel_of_Adun"))
        buildingsTypesSeen.insert(Protoss_Citadel_of_Adun);
    else if (ut.compare("Protoss_Cybernetics_Core"))
        buildingsTypesSeen.insert(Protoss_Cybernetics_Core);
    else if (ut.compare("Protoss_Templar_Archives"))
        buildingsTypesSeen.insert(Protoss_Templar_Archives);
    else if (ut.compare("Protoss_Forge"))
        buildingsTypesSeen.insert(Protoss_Forge);
    else if (ut.compare("Protoss_Stargate"))
        buildingsTypesSeen.insert(Protoss_Stargate);
    else if (ut.compare("Protoss_Fleet_Beacon"))
        buildingsTypesSeen.insert(Protoss_Fleet_Beacon);
    else if (ut.compare("Protoss_Arbiter_Tribunal"))
        buildingsTypesSeen.insert(Protoss_Arbiter_Tribunal);
    else if (ut.compare("Protoss_Robotics_Support_Bay"))
        buildingsTypesSeen.insert(Protoss_Robotics_Support_Bay);
    else if (ut.compare("Protoss_Shield_Battery"))
        buildingsTypesSeen.insert(Protoss_Shield_Battery);
    
    /// Terran
    if (ut.compare("Terran_Command_Center"))
    {
        int tmp = Terran_Command_Center;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Terran_Command_Center3)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Terran_Comsat_Station"))
        buildingsTypesSeen.insert(Terran_ComSat);
    else if (ut.compare("Terran_Nuclear_Silo"))
        buildingsTypesSeen.insert(Terran_Nuclear_Silo);
    else if (ut.compare("Terran_Supply_Depot"))
    {
        int tmp = Terran_Supply_Depot;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Terran_Supply_Depot3)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Terran_Refinery"))
    {
        int tmp = Terran_Refinery;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Terran_Refinery2)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Terran_Barracks"))
    {
        int tmp = Terran_Barracks;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Terran_Barracks4)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Terran_Academy"))
        buildingsTypesSeen.insert(Terran_Academy);
    else if (ut.compare("Terran_Factory"))
        buildingsTypesSeen.insert(Terran_Factory);
    else if (ut.compare("Terran_Starport"))
        buildingsTypesSeen.insert(Terran_Starport);
    else if (ut.compare("Terran_Control_Tower"))
        buildingsTypesSeen.insert(Terran_Control_Tower);
    else if (ut.compare("Terran_Science_Facility"))
        buildingsTypesSeen.insert(Terran_Science_Facility);
    else if (ut.compare("Terran_Covert_Ops"))
        buildingsTypesSeen.insert(Terran_Covert_Ops);
    else if (ut.compare("Terran_Physics_Lab"))
        buildingsTypesSeen.insert(Terran_Physics_Lab);
    else if (ut.compare("Terran_Machine_Shop"))
        buildingsTypesSeen.insert(Terran_Machine_Shop);
    else if (ut.compare("Terran_Engineering_Bay"))
        buildingsTypesSeen.insert(Terran_Engineering_Bay);
    else if (ut.compare("Terran_Armory"))
        buildingsTypesSeen.insert(Terran_Armory);
    else if (ut.compare("Terran_Missile_Turret"))
        buildingsTypesSeen.insert(Terran_Missile_Turret);
    else if (ut.compare("Terran_Bunker"))
        buildingsTypesSeen.insert(Terran_Bunker);

    /// Zerg
    else if (ut.compare("Zerg_Hatchery"))
    {
        int tmp = Zerg_Hatchery;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Zerg_Hatchery4)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Zerg_Lair"))
        buildingsTypesSeen.insert(Zerg_Lair);
    else if (ut.compare("Zerg_Hive"))
        buildingsTypesSeen.insert(Zerg_Hive);
    else if (ut.compare("Zerg_Nydus_Canal"))
        buildingsTypesSeen.insert(Zerg_Nydus_Canal);
    else if (ut.compare("Zerg_Hydralisk_Den"))
        buildingsTypesSeen.insert(Zerg_Hydralisk_Den);
    else if (ut.compare("Zerg_Defiler_Mound"))
        buildingsTypesSeen.insert(Zerg_Defiler_Mound);
    else if (ut.compare("Zerg_Greater_Spire"))
        buildingsTypesSeen.insert(Zerg_Greater_Spire);
    else if (ut.compare("Zerg_Queens_Nest"))
        buildingsTypesSeen.insert(Zerg_Queens_Nest);
    else if (ut.compare("Zerg_Evolution_Chamber"))
        buildingsTypesSeen.insert(Zerg_Evolution_Chamber);
    else if (ut.compare("Zerg_Ultralisk_Cavern"))
        buildingsTypesSeen.insert(Zerg_Ultralisk_Cavern);
    else if (ut.compare("Zerg_Spire"))
        buildingsTypesSeen.insert(Zerg_Spire);
    else if (ut.compare("Zerg_Spawning_Pool"))
        buildingsTypesSeen.insert(Zerg_Spawning_Pool);
    else if (ut.compare("Zerg_Creep_Colony"))
        buildingsTypesSeen.insert(Zerg_Creep_Colony);
    else if (ut.compare("Zerg_Spore_Colony"))
        buildingsTypesSeen.insert(Zerg_Spore_Colony);
    else if (ut.compare("Zerg_Sunken_Colony"))
        buildingsTypesSeen.insert(Zerg_Sunken_Colony);
    else if (ut.compare("Zerg_Extractor"))
    {
        int tmp = Zerg_Extractor;
        while (buildingsTypesSeen.count(tmp))
            ++tmp;
        if (tmp <= Zerg_Extractor2)
            buildingsTypesSeen.insert(tmp);
    }
    else if (ut.compare("Zerg_Overlord"))
    {
        if (notFirstOverlord)
        {
            int tmp = Zerg_Overlord;
            while (buildingsTypesSeen.count(tmp))
                ++tmp;
            if (tmp <= Zerg_Overlord3)
                buildingsTypesSeen.insert(tmp);
        }
        else
            notFirstOverlord = true;
    }

    if (buildingsTypesSeen.size() > previous_size)
        return true;
    else
        return false;
}

void tech_estimator::computeDistribOpenings(int time)
{
    if (time >= LEARNED_TIME_LIMIT || time <= 0)
        return;

    size_t nbXes = st.vector_X.size();
    list<unsigned int> compatibleXes;
    for (size_t i = 0; i < nbXes; ++i)
    {
        if (testBuildTreePossible(i, buildingsTypesSeen))
            compatibleXes.push_back(i);
    }
    long double runningSum = 0.0;
    for (size_t i = 0; i < openingsProbas.size(); ++i)
    {
        long double sumX = MIN_PROB;
        for (list<unsigned int>::const_iterator it = compatibleXes.begin();
                it != compatibleXes.end(); ++it)
        { /// perhaps underflow? log-prob?
            sumX += st.tabulated_P_X_Op[(*it) * openingsProbas.size() + i]
                * st.tabulated_P_Time_X_Op[(*it) * openingsProbas.size() * LEARNED_TIME_LIMIT
                + i * LEARNED_TIME_LIMIT + time];
        }
        openingsProbas[i] *= sumX;
        runningSum += openingsProbas[i];
    }
    long double verifSum = 0.0;
    for (size_t i = 0; i < openingsProbas.size(); ++i)
    {
        openingsProbas[i] /= runningSum;
        if (openingsProbas[i] != openingsProbas[i] // test for NaN / 1#IND
                //|| openingsProbas[i] < MIN_PROB        // min proba
           )
            openingsProbas[i] = MIN_PROB;
        verifSum += openingsProbas[i];
    }
    if (verifSum < 0.99 || verifSum > 1.01)
    {
        for (size_t i = 0; i < openingsProbas.size(); ++i)
            openingsProbas[i] /= verifSum;
    }
    hasInfered = true;
}

/**
 * Tests if the given build tree (X) value
 * is compatible with obervations (what have been seen)
 * {X ^ observed} covers all observed if X is possible
 * so X is impossible if {observed \ {X ^ observed}} != {}
 * => X is compatible with observations if it covers them fully
 */
bool tech_estimator::testBuildTreePossible(int indBuildTree, const set<int>& setObs)
{
    for (set<int>::const_iterator it = setObs.begin();
            it != setObs.end(); ++it)
    {
        if (!st.vector_X[indBuildTree].count(*it))
            return false;
    }
    return true;
}

const std::vector<long double>& tech_estimator::getOpeningsProbas() const
{
    return openingsProbas;
}

void tech_estimator::dump_vector_X()
{
    for (vector<set<int> >::const_iterator tt = st.vector_X.begin();
            tt != st.vector_X.end(); ++tt)
    {
        for (set<int>::const_iterator building = tt->begin();
                building != tt->end(); ++building)
            cout << *building << " ";
        cout << endl;
    }
}

int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        cout << "usage:" << endl;
        cout << "./tech_estimator XvY" << endl;
        cout << "where XvY is a match-up (X and Y in {P,T,Z})" << endl;
        cout << "Assuming existence of tables/{gauss|laplace}/*.table" << endl;
        return -1;
    }
    tech_estimator t_e = tech_estimator(string(argv[1]));
#ifdef DUMP_ENUM_
    for (int i = 0; i < NB_TERRAN_BUILDINGS; ++i)
        cout << terran_buildings_name[i] << endl;
    for (int i = 0; i < NB_PROTOSS_BUILDINGS; ++i)
        cout << protoss_buildings_name[i] << endl;
    for (int i = 0; i < NB_ZERG_BUILDINGS; ++i)
        cout << zerg_buildings_name[i] << endl;
#endif
#ifdef DUMP_VECTOR_X
    t_e.dump_vector_X();
#endif
    /*
    TODO
    while (qqch sur stdin)
    {
        t_e.onUnitShow(qqch.unit_name, qqch.frame);
        t_e.outputTTDistrib();
    }
    */
    return 0;
}
