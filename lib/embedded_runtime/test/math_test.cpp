#include "registers.h"
#include "serial.h"
#include "pll_driver.h"
#include "heap.h"
#include "StdFunctions.h"
#include "Print.h"

PLL_Driver pll;


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
    pll.setMaxFrequency();

    RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;
	GPIOA->MODER |= 1<<10;
    SerialPort serial(115200);
        
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

        while (h->next != nullptr){
            serial.write(h->data);
            serial.write("\n\r");
            h = h->next;
        }
        
    for(;;){
    }
}