#ifndef MARCO_RUNTIME_DRIVERS_IDA_DRIVER_H
#define MARCO_RUNTIME_DRIVERS_IDA_DRIVER_H

#include "marco/Runtime/Drivers/Driver.h"

namespace marco::runtime
{
  class IDA : public Driver
  {
    public:
      IDA(Simulation* simulation);

#ifdef CLI_ENABLE
      std::unique_ptr<cli::Category> getCLIOptions() override;
#endif // CLI_ENABLE

      int run() override;
  };
}

//===---------------------------------------------------------------------===//
// Functions defined inside the module of the compiled model
//===---------------------------------------------------------------------===//

extern "C"
{
  void* initICSolvers();
  void* deinitICSolvers();

  void solveICModel();

  void initMainSolvers();
  void deinitMainSolvers();

  void calcIC();

  void updateIDAVariables();
  void updateNonIDAVariables();
  double getIDATime();
}

#endif // MARCO_RUNTIME_DRIVERS_IDA_DRIVER_H
