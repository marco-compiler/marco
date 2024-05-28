#ifndef MARCO_RUNTIME_SIMULATION_OPTIONS_H
#define MARCO_RUNTIME_SIMULATION_OPTIONS_H

#include <cstdint>

namespace marco::runtime::simulation
{
  struct Options
  {
    bool debug = false;

    double startTime = 0;
    double endTime = 10;

    // The number of partitions each thread should process in an ideal scenario
    // in which all the equations are independent from each other and have
    // equal computational cost.
    int64_t equationsPartitioningFactor = 10;

    // The number of steps to be executed by the scheduler, first in sequential
    // and then multithreaded mode, to decide on the execution strategy of the
    // scheduler itself.
    int64_t schedulerCalibrationRuns = 10;
  };

  Options& getOptions();
}

#endif // MARCO_RUNTIME_SIMULATION_OPTIONS_H
