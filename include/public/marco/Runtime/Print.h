#ifndef MARCO_RUNTIME_PRINT_H
#define MARCO_RUNTIME_PRINT_H

#include "marco/Runtime/ArrayDescriptor.h"
#include "marco/Runtime/Mangling.h"
#include <cstdint>

class PrinterConfig
{
  public:
    bool scientificNotation = false;
    unsigned int precision = 9;
};

PrinterConfig& printerConfig();

RUNTIME_FUNC_DECL(print, void, bool)
RUNTIME_FUNC_DECL(print, void, int32_t)
RUNTIME_FUNC_DECL(print, void, int64_t)
RUNTIME_FUNC_DECL(print, void, float)
RUNTIME_FUNC_DECL(print, void, double)

RUNTIME_FUNC_DECL(print, void, ARRAY(bool))
RUNTIME_FUNC_DECL(print, void, ARRAY(int32_t))
RUNTIME_FUNC_DECL(print, void, ARRAY(int64_t))
RUNTIME_FUNC_DECL(print, void, ARRAY(float))
RUNTIME_FUNC_DECL(print, void, ARRAY(double))

RUNTIME_FUNC_DECL(print_csv_newline, void)
RUNTIME_FUNC_DECL(print_csv_separator, void)
RUNTIME_FUNC_DECL(print_csv_name, void, PTR(void), int64_t, PTR(int64_t))

RUNTIME_FUNC_DECL(print_csv, void, bool)
RUNTIME_FUNC_DECL(print_csv, void, int32_t)
RUNTIME_FUNC_DECL(print_csv, void, int64_t)
RUNTIME_FUNC_DECL(print_csv, void, float)
RUNTIME_FUNC_DECL(print_csv, void, double)

#endif // MARCO_RUNTIME_PRINT_H
