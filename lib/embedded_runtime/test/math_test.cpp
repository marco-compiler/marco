#include "../include/marco/driver/registers.h"
#include "../include/marco/driver/serial.h"
#include "../include/marco/driver/pll_driver.h"
#include "../include/marco/driver/heap.h"
#include "../include/marco/lib/StdFunctions.h"
#include "../include/marco/lib/Print.h"
#include <initializer_list>

PLL_Driver pll;

SerialPort serial(115200);

void f(std::initializer_list<int> const &items){
    print_char("CIAO");
};

void delay()
{
	volatile int i;
	for(i=0;i<1000000;i++) ;
}

void foo(){
    GPIOA->BSRR=1<<5;
}
void bar(){
    GPIOA->BSRR=1<<(5 + 16);
}


int main(){
    

    RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;
	GPIOA->MODER |= 1<<10;

    //int a = stde::abs(3);
        /*
        foo();
        serial.write("log of 10 =>");
        serial.write(stde::log(10));
        serial.write("\n\r");
        delay();
        bar();
        
        foo();
        serial.write("log di 0.2 => ");
        serial.write(stde::log(0.2),4);
        serial.write("\n\r");
        delay();
        bar();

        foo();
        serial.write("exp di 0.5 =>");
        serial.write(stde::exp(.5),3);
        serial.write("\n\r");
        delay();
        bar();

          foo();
        serial.write("cosh di 15=>");
        serial.write(stde::cosh(14));
        serial.write("\n\r");
        delay();
        bar();
        foo();
        serial.write("cosh di 0.75 =>");
        serial.write(stde::cosh(.75));
        serial.write("\n\r");
        delay();
        bar();

        serial.write(8103.0839f);
        serial.write("\n\r");
        serial.write(stde::sqrt(2.3),5);
        serial.write("\n\r");
        serial.write(stde::sqrt(0.75),5);
        serial.write("\n\r");
        serial.write(14300.034f);
        serial.write("\n\r VECTOR \n\r");
        stde::Vector<int> v;
        int a = 302;
        v.push_back(a);
        int b = v.last->data;
        v.push_back(12);
        serial.write("second last \n\r");
        serial.write(v.second_last->data);
        serial.write("\n\r");
        v.push_back(539);
        serial.write("second last \n\r");
        serial.write(v.second_last->data);
         serial.write("\n\r");
        stde::Vector<int>::Node* h;
        h = v.head;

        stde::array<float,20> arr;  

        int init[3] = {2,3,4};
        print_char("start \n\r");
        print_integer_array(init);
        print_char("Array_end  \n\r");
        int* value = (int*) malloc(sizeof(int));
        *value = 134;
        serial.write("Before ");
        print_integer(*value);
        serial.write("\n\r");
        //int* new_value = stde::forward<int>(value);
        //free(value);
        serial.write("After ");
        print_serial(*value);
        serial.write("\n\r");
        //serial.write(*new_value);

        print_serial(1.234f);
        */
       //f({1,2,3,4});
       /*
       stde::Vector<int> v = {2,3,4,5};
       v.push_back(2);
       v.push_back(3);
       
       for(int i :v){
           print_integer(i);
           print_char("\n\r");
       }

        print_char("Size : ");
        print_integer(v.size());
        print_char("\n\r");
    
       stde::array<float,4> farr = {3.3,3.1,22,12};
       *
       // stde::array<int,3> farr = {2,3,4};
        /*
       print_float(farr.size());
       print_char("\n\rBegin ");
       print_float(*farr.begin());
       print_char("\n\r End ");
       print_float(*farr.end());
       print_char("\n\r");

       for(float f : farr){
        //for(int i = 0; i < farr.size();i++){
           print_float(f);
           //print_float(farr[i]);
           print_char("<>\n\r");
       }
       //print_integer(farr.max_size());
        //print_array(array);
    */
   /*
        int* a = (int*) malloc(sizeof(int));
        *a = 12;
        serial.write("\n\r Before ");
        serial.write(*a);

        free(a);

        serial.write("\n\r After ");
        serial.write(*a);
         
        int* b = (int*) malloc(sizeof(int));
        *b = 123;
        serial.write("\n\r New Alloc ");
        serial.write(*b);
        serial.write("\n\r After ");
        serial.write(*a);
        //serial.write("CIAO");
    
       
        for(float f : farr){
        //for(int i = 0; i < farr.size();i++){
           print_char("\n\r<>");
           print_float(f);
           //print_float(farr[i]);
           print_char("<>\n\r");
       }
       */

      long int g = 1;
      print_char("CIAO\n\r");
      printf("PROVA %c [%d] ;;;%12f",'c', 1 , 0.231454253242);
      printf("\n\rend");
    
    for(;;){
    }
}