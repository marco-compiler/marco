#include <iostream>
#include <vector>
#include "../include/marco/lib/StdFunctions.h"

void f(std::initializer_list<int> list){
    stde::Vector<int> vec;
    for( int i : list){
        vec.push_back(i);
    }
    std::cout <<"Vector begin " << std::begin(vec)[0];
}


int main(int argc, char const *argv[])
{   
    stde::Vector<int> vector  {2,4,5};
    vector.push_back(22);
    vector.push_back(3);
    vector.push_back(4);
    auto prova = std::begin(vector);
    std::cout << "BEGIN " << prova[0] <<std::endl;
    f({3,4,6,8,9});
    return 0;
}
