#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "army_gmm.h"

using namespace std;

int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        cout << "usage: ./load_gmm FILE" << endl;
        cout << "for instance: ./load_gmm Pv_P_.gmm" << endl;
        return -1;
    }
    cout << endl;
    cout << "Parsing " << argv[1] << endl;
    cout << endl;

    /// PARSE
    ifstream ifs(argv[1]);
    string line;
    bool order = false;
    bool means = false;
    bool covars = false;
    vector<string> tab_order;
    vector<vector<long double> > tab_mu;
    vector<vector<vector<long double> > > tab_sigma;
    //while (ifs.good())
    vector<vector<long double> > tmp_sigma;
    while (std::getline(ifs, line))
    {
        //cout << line << endl;
        stringstream linestream(line);
        string word1;
        linestream >> word1;
        //cout << word1 << endl;
        if (word1[0] == 'n') // "n ***"
            ;
        else if (word1[0] == 'i') // "_i_n order" order of the unit types
            order = true;
        else if (word1[0] == 'm') // means
            means = true;
        else if (word1[0] == 'c' && word1[2] == 'v') // covars
            covars = true;
        else if (word1[0] == 'f') // feature
            ;
        else if (word1[0] == 'c' && word1[2] == 'm') // component
        {
            if (covars)// && !tmp_sigma.empty())
            {
                tab_sigma.push_back(tmp_sigma);
                tmp_sigma.clear();
            }
        }
        else
        {
            string lastWord = "";
            vector<long double> linevec;
            vector<string> linevec_s;
            while (word1.compare(lastWord))
            {
                //cout << word1 << endl;
                if (means || covars)
                    linevec.push_back(atof(word1.c_str()));
                else
                    linevec_s.push_back(word1);
                lastWord = word1;
                linestream >> word1;
            }
            if (covars)
                tmp_sigma.push_back(linevec);
            else if (means)
                tab_mu.push_back(linevec);
            else if (order)
                tab_order.swap(linevec_s);
        }
    }
    ifs.close();
    
    /// PRINT table (to compare with the text file)
    cout << "*** Order ***" << endl;
    for (vector<string>::const_iterator it = tab_order.begin();
            it != tab_order.end(); ++it)
        cout << *it << " ";
    cout << endl;
    cout << "=============" << endl;
    cout << "*** Mu ***" << endl;
    for (vector<vector<long double> >::const_iterator it = tab_mu.begin();
            it != tab_mu.end(); ++it)
    {
        for (vector<long double>::const_iterator it2 = it->begin();
                it2 != it->end(); ++it2)
            cout << *it2 << " ";
        cout << endl;
    }
    cout << "=============" << endl;
    cout << "*** Sigma ***" << endl;
    for (vector<vector<vector<long double> > >::const_iterator it = tab_sigma.begin();
            it != tab_sigma.end(); ++it)
    {
        for (vector<vector<long double> >::const_iterator it2 = it->begin();
                it2 != it->end(); ++it2)
        {
            for (vector<long double>::const_iterator it3 = it2->begin();
                    it3 != it2->end(); ++it3)
                cout << *it3 << " ";
            cout << endl;
        }
        cout << endl;
    }
    
    /// SERIALIZE
    army_gmm gmm;
    gmm.order.swap(tab_order);
    gmm.mu.swap(tab_mu);
    gmm.sigma.swap(tab_sigma);
    string filename(argv[1]);
    filename.append("b");
    {
        ofstream writefile(filename.c_str());
        if (writefile.good())
        {
            boost::archive::text_oarchive archive(writefile);
            archive << gmm;
        }
        else
            return -1;
    }

    /// PRINT SERIALIZED
    cout << "SERIALIZED:" << endl;
    army_gmm gmm2;
    ifstream ifs2(filename.c_str());
    boost::archive::text_iarchive ia(ifs2);
    ia >> gmm2;
    cout << "*** Order ***" << endl;
    for (vector<string>::const_iterator it = gmm2.order.begin();
            it != gmm2.order.end(); ++it)
        cout << *it << " ";
    cout << endl;
    cout << "=============" << endl;
    cout << "*** Mu ***" << endl;
    for (vector<vector<long double> >::const_iterator it = gmm2.mu.begin();
            it != gmm2.mu.end(); ++it)
    {
        for (vector<long double>::const_iterator it2 = it->begin();
                it2 != it->end(); ++it2)
        {
            cout << *it2 << " ";
        }
        cout << endl;
    }
    cout << "=============" << endl;
    cout << "*** Sigma ***" << endl;
    for (vector<vector<vector<long double> > >::const_iterator it = gmm2.sigma.begin();
            it != gmm2.sigma.end(); ++it)
    {
        for (vector<vector<long double> >::const_iterator it2 = it->begin();
                it2 != it->end(); ++it2)
        {
            for (vector<long double>::const_iterator it3 = it2->begin();
                    it3 != it2->end(); ++it3)
                cout << *it3 << " ";
            cout << endl;
        }
        cout << endl;
    }

    return 0;
}
