#ifndef MARCO_DIAGNOSTIC_LEVEL_H
#define MARCO_DIAGNOSTIC_LEVEL_H

namespace marco::diagnostic
{
  enum class Level
  {
    FATAL_ERROR,
    ERROR,
    WARNING,
    REMARK,
    NOTE
  };
}

#endif // MARCO_DIAGNOSTIC_LEVEL_H
