#include <iostream>

template<class T>
bool is_integral(T data){
        return dynamic_cast<int*>(data) != nullptr;
    }


int main(int argc, char const *argv[])
{
    int *g;
    *g = 30;
    //std::cout << is_integral(g);
    int *a;
    *a = 203;
    int* b = dynamic_cast<float*>(a);
    return 0;
}
