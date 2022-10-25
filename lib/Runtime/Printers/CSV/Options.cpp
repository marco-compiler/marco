#include "marco/Runtime/Printers/CSV/Options.h"

namespace marco::runtime::printing
{
  PrintOptions& printOptions()
  {
    static PrintOptions obj;
    return obj;
  }
}
