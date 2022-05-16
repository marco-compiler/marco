#include "../../include/marco/lib/Print.h"
#include "../../include/marco/driver/serial.h"

extern SerialPort serial;

void print_char(const char *s){
    serial.write(s);
}

void print_integer(const int n){
    serial.write(n);
};

void print_float(const float f){
    serial.write(f);
};  

void print_float_precision(const float f, const int p){
    serial.write(f,p);
};

void print_float_array(const float array[]){
    for(int i = 0; i < sizeof(array)/sizeof(array[0]); i++){
        serial.write(array[i]);
        serial.write(" ");
    }
    serial.write("\n\r");
}

void print_integer_array(const int array[]){
    for(int i = 0; i < sizeof(array)/sizeof(array[0]); i++){
        serial.write(array[i]);
        serial.write(" ");
    }
    serial.write("\n\r");
}



