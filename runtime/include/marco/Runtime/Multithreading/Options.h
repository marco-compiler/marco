#ifndef MARCO_RUNTIME_MULTITHREADING_OPTIONS_H
#define MARCO_RUNTIME_MULTITHREADING_OPTIONS_H

#ifdef THREADS_ENABLE

#include <thread>

namespace marco::runtime::multithreading
{
  struct MultithreadingOptions
  {
    bool enableMultithreading = true;
    unsigned int numOfThreads = std::thread::hardware_concurrency();
  };

  MultithreadingOptions& multithreadingOptions();
}

#endif // THREADS_ENABLE

#endif // MARCO_RUNTIME_MULTITHREADING_OPTIONS_H
