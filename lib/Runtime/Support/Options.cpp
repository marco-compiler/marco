#include "marco/Runtime/Support/Options.h"

namespace marco::runtime::support
{
  SupportOptions& supportOptions()
  {
    static SupportOptions obj;
    return obj;
  }
}
