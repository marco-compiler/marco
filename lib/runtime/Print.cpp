#include "marco/runtime/Print.h"
#include "../embedded_runtime/include/marco/driver/serial.h"

extern SerialPort serial;

void Print::print_char(const char *s){
    serial.write(s);
}

void Print::print_integer(const int n){
    serial.write(n);
};

void Print::print_float(const float f){
    serial.write(f);
};  

void Print::print_float_precision(const float f, const int p){
    serial.write(f,p);
};