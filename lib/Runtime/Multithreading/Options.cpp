#include "marco/Runtime/Multithreading/Options.h"

namespace marco::runtime::multithreading
{
  MultithreadingOptions& multithreadingOptions()
  {
    static MultithreadingOptions obj;
    return obj;
  }
}
