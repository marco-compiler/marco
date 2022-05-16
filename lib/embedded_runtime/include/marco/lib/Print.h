#ifndef PRINT_H
#define PRINT_H

#include "../driver/serial.h"

extern SerialPort serial;

void print_char(const char *s);
    
void print_integer(const int n);
    
void print_float(const float f);
    
void print_float_precision(const float f, const int p);
    
void print_integer_array(const int array[]);

void print_float_array(const float array[]);



template<class T>
void print_serial(T value){
    serial.write(value);
}

template<class T>
void print_array(T *array){
    for(int i = 0; i < sizeof(array)/sizeof(array[0]);i++){
        serial.write(*(array + i));
        serial.write(" ");
    }
    serial.write("\n\r");
}
#endif