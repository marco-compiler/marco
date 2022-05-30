#include <iostream>
#include <stdarg.h>
#include <cstdint>

enum class byte_t : unsigned char {};

void floatToBytes(byte_t* bytes, float flt)
{
  bytes[1] = (byte_t) flt;    //truncate whole numbers
  flt = (flt - bytes[1])*100; //remove whole part of flt and shift 2 places over
  bytes[0] = (byte_t) flt;    //truncate the fractional part from the new "whole" part
}


int main(int argc, char const *argv[])
{   
    char a = '9';
    std::cout << a << std::endl;

    std::cout << (int)a - 48 << std::endl;
    return 0;
}
