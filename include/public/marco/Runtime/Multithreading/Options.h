#ifndef MARCO_RUNTIME_MULTITHREADING_OPTIONS_H
#define MARCO_RUNTIME_MULTITHREADING_OPTIONS_H

#include <thread>

namespace marco::runtime::multithreading
{
  struct MultithreadingOptions
  {
    unsigned int numOfThreads = std::thread::hardware_concurrency();
  };

  MultithreadingOptions& multithreadingOptions();
}

#endif // MARCO_RUNTIME_MULTITHREADING_OPTIONS_H
