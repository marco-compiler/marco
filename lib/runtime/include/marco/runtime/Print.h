#ifndef PRINT_H
#define PRINT_H

#include "../embedded_runtime/include/marco/driver/serial.h"
//#include "../embedded_runtime/include/marco/driver/heap.h"

void print_char(const char *s);

void print_integer(const int n);

void print_float(const float f);

void print_float_precision(const float f, const int p);

#endif