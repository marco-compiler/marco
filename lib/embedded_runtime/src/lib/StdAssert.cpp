#include "../../include/marco/lib/StdFunctions.h"


extern void UnimplementedIrq();

void stde::assertt(bool cond){
    if(cond)
    return;
    else
    while(true);
}
