#include "../include/marco/driver/registers.h"
#include "../include/marco/driver/serial.h"
#include "../include/marco/driver/pll_driver.h"
#include "../include/marco/driver/heap.h"
#include "../include/marco/lib/Runtime.h"
#include "../include/marco/lib/Print.h"

PLL_Driver pll;
extern void runSimulation();
SerialPort serial(115200);

int main(){

    print_char("Start Simulation\n\r");
    runSimulation();
    serial.write("\n\rEnd Simulation\n\r\0");
    for(;;){

    }
}
