#include "../../include/marco/lib/StdFunctions.h"


extern void UnimplementedIrq();

void stde::assertt(bool cond){
    if(cond)
    return;
    else
    while(true);//UnimplementedIrq(); //send to an interrupt in case assertion is not satisfied.
}
