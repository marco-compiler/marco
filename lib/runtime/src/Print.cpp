#include "Print.h"

SerialPort serial;

void print_char(const char *s){
    serial.write(s);
};

void print_integer(const int n){
    serial.write(n);
};

void print_float(const float f){
    serial.write(f);
};  

void print_float_precision(const float f, const int p){
    serial.write(f,p);
};