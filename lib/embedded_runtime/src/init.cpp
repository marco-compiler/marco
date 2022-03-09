#include "registers.h"
#include "serial.h"
#include "pll_driver.h"
#include "heap.h"
#include "marco/runtime/Runtime.h"


PLL_Driver pll;
extern void runSimulation();

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
    SerialPort serial(115200);
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;
	GPIOA->MODER |= 1<<10;
    foo();
    serial.write("Start Simulation\n\r");
    runSimulation();
    serial.write("End Simulation\n\r");
    bar();
    for(;;){
        foo();
        delay();
        bar();
        delay();
    }
}