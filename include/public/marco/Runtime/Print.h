#ifndef MARCO_RUNTIME_PRINT_H
#define MARCO_RUNTIME_PRINT_H

#include "marco/Runtime/Mangling.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>

#ifdef MARCO_CLI

#include "marco/Runtime/CLI.h"

namespace marco::runtime::formatting
{
  std::unique_ptr<cli::Category> getCLIOptionsCategory();
}

#endif

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
