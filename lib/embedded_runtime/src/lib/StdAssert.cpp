#include "../../include/marco/lib/StdFunctions.h"


extern void UnimplementedIrq();



//CASSERT
void stde::assert(bool cond){
    if(cond)
    return;
    else
    UnimplementedIrq(); //send to an interrupt in case assertion is not satisfied.
}
