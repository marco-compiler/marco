#include "../../include/marco/lib/Print.h"
#include "../../include/marco/driver/serial.h"
#include <stdarg.h>

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


void printf(char* string){
    serial.write(string);
}

int getPrecision(int* values, int digits){
    int precision = 0;
    int m[3] = {100,10,1};
    int limit = digits <= 3 ? digits-1 : 2;
    for(int i = limit; i >= 0;i--){
        precision += values[i] * m[i];
    }
    return precision;
}

int printf(const char* string, ...){
    va_list lst;
    va_start(lst, string);
    


    while(*string != '\0')
    {   //serial.write("while");
        if(*string != '%')
        {   //char  = *string;
            serial.write(string[0]);
            string++;
            continue;
        }

        string++;
        /*
        if(*string == '\0')
        {
            break;
        }
        int *values;
        int digits = 0;
        if(*string == 'd'){
            int d = va_arg(lst,int);
            serial.write(d);
        }if(*string > 48 && *string <=57){
            while(*string != 'f'){
                *(values + digits++) = (int)*string;
                if(!(*string > 48 && *string <=57) && *string != 'f') break;
                string++;
                if(*string == 'f'){
                    int precision = getPrecision(values,digits);
                    serial.write(va_arg(lst,float),precision);
                }
            }
        }else if(*string == 'f') serial.write(va_arg(lst,float));
        */
        char c;
        int d;
        float f;
        int *values;
        int digits = 0;
        int precision  = 0;
        switch(*string)
        {
            //case 's': fputs(va_arg(lst, char *), stdout); break;
            case 'c':
                c = va_arg(lst,int);
                serial.write(c);
                break;
            case 'd':
                d = va_arg(lst,int);
                serial.write(d);
                break;
            case 'f':
                f = va_arg(lst,double);
                serial.write(f);
                break;
            default:
                while((*string >= 48 && *string <=57) && *string != 'f'){
                    char t = string[0];
                    *(values + digits) = (int)t - 48;
                    digits++;
                    if(!(*string >= 48 && *string <=57) && *string != 'f') break;
                    string++;
                    if(*string == 'f'){
                        if(digits == 1) precision = (int)t - 48 ;//?
                        else precision = getPrecision(values,digits);
                        serial.write("\n\r PRECISION = ");
                        serial.write(precision);
                        serial.write("\n\r");
                        serial.write((float)va_arg(lst,double),precision);
                }
                }
            break;
            
        }
        string++;
    }
    return 0;
}