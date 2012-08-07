#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "c_knowing_ec_table.h"

using namespace std;

int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        cout << "usage: ./load_cec FILE" << endl;
        cout << "for instance: ./load_cec PvP.cec" << endl;
        return -1;
    }
    cout << endl;
    cout << "Parsing " << argv[1] << endl;
    cout << endl;

    /// PARSE
    ifstream ifs(argv[1]);
    string line;
    vector<vector<long double> > table;
    //while (ifs.good())
    while (std::getline(ifs, line))
    {
        //cout << line << endl;
        stringstream linestream(line);
        string word1;
        linestream >> word1;
        //cout << word1 << endl;
        if (word1[0] == 'P' || word1[0] == 'Z' || word1[0] == 'T') 
            ; // first line (doc)
        else if (word1[0] == 'c' && word1[2] == 'm') // component
            ;
        else
        {
            string lastWord = "";
            vector<long double> linevec;
            while (word1.compare(lastWord))
            {
                //cout << word1 << endl;
                linevec.push_back(atof(word1.c_str()));
                lastWord = word1;
                linestream >> word1;
            }
            table.push_back(linevec);
        }
    }
    ifs.close();
    
    /// PRINT table (to compare with the text file)
    for (vector<vector<long double> >::const_iterator it = table.begin();
            it != table.end(); ++it)
    {
        for (vector<long double>::const_iterator it2 = it->begin();
                it2 != it->end(); ++it2)
        {
            cout << *it2 << " ";
        }
        cout << endl;
    }
    
    /// SERIALIZE
    c_knowing_ec_table st;
    st.cec.swap(table);
    string filename(argv[1]);
    filename.append("b");
    {
        ofstream writefile(filename.c_str());
        if (writefile.good())
        {
            boost::archive::text_oarchive archive(writefile);
            archive << st;
        }
        else
            return -1;
    }

    /// PRINT SERIALIZED
    cout << "SERIALIZED:" << endl;
    c_knowing_ec_table st2;
    ifstream ifs2(filename.c_str());
    boost::archive::text_iarchive ia(ifs2);
    ia >> st2;
    for (vector<vector<long double> >::const_iterator it = st2.cec.begin();
            it != st2.cec.end(); ++it)
    {
        for (vector<long double>::const_iterator it2 = it->begin();
                it2 != it->end(); ++it2)
        {
            cout << *it2 << " ";
        }
        cout << endl;
    }

    return 0;
}
