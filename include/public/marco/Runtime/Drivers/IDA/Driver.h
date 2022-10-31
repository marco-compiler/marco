#ifndef MARCO_RUNTIME_DRIVERS_IDA_DRIVER_H
#define MARCO_RUNTIME_DRIVERS_IDA_DRIVER_H

#include "marco/Runtime/Drivers/Driver.h"

namespace marco::runtime
{
  class IDA : public Driver
  {
    public:
      IDA(Simulation* simulation);

      std::unique_ptr<cli::Category> getCLIOptions() override;

      int run() override;
  };
}

//===---------------------------------------------------------------------===//
// Functions defined inside the module of the compiled model
//===---------------------------------------------------------------------===//

extern "C"
{
  void* initICSolvers(void* data);
  void* deinitICSolvers(void* data);

  void solveICModel(void* data);

  void initMainSolvers(void* data);
  void deinitMainSolvers(void* data);

  void calcIC(void* data);

  void updateNonStateVariables(void* data);
  void updateStateVariables(void* data);
  void incrementTime(void* data);
}

#endif // MARCO_RUNTIME_DRIVERS_IDA_DRIVER_H
