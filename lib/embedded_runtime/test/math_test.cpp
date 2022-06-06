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

    printf("Cos: %8f \n\r", stde::cos(13));
    printf("Sin: %8f", stde::sin(13));
    long int g = 1;
    print_char("CIAO\n\r");
    printf("PROVA %c [%d] ;;;%12f",'c', 1 , 0.334);
    printf("\n\rend");
    
    for(;;){
    }
}

