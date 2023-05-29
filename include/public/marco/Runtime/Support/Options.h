#ifndef MARCO_RUNTIME_SUPPORT_OPTIONS_H
#define MARCO_RUNTIME_SUPPORT_OPTIONS_H

#include <cstdint>

namespace marco::runtime::support
{
  struct SupportOptions
  {
    bool useSinInterpolation = false;
    int64_t sinInterpolationPoints = 10;

    bool useCosInterpolation = false;
    int64_t cosInterpolationPoints = 10;
  };

  SupportOptions& supportOptions();
}

#endif // MARCO_RUNTIME_SUPPORT_OPTIONS_H
