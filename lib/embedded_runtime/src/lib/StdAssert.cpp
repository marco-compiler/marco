#include "../../include/marco/lib/StdFunctions.h"


extern void UnimplementedIrq();

void stde::assertt(bool cond){
    if(cond)
    return;
    else
    return;//UnimplementedIrq(); //send to an interrupt in case assertion is not satisfied.
}
