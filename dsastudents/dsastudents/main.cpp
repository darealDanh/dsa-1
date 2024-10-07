#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
// #include "list/listheader.h"
#include "ann/dataloader.h"
#include "ann/dataset.h"

using namespace std;

int main(int argc, char **argv)
{
    string name = "data12";
    //! data
    int nsamples = 100;
    xt::xarray<float> data = {1, 2, 3, 4};
    xt::xarray<string> label = {"one", "two", "three", "four"};
    TensorDataset<double, string> ds(data, label);
    DataLoader<double, string> loader(&ds, 3, false, true);
    //! expect
    string expect = "getData={ 1.,  2.,  3.} ; getLabel={  one,   two, three} ";

    //! output

    stringstream output;
    output << "getData=";
    for (auto it = loader.begin(); it != loader.end(); ++it)
    {
        output << (*it).getData() << " ";
    }
    output << "; getLabel=";
    for (auto it = loader.begin(); it != loader.end(); ++it)
    {
        output << (*it).getLabel() << " ";
    }
    cout << output.str() << endl;
    cout << "Expect: " << expect << endl;
    return 0;
}
