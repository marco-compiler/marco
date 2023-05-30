#ifndef MARCO_RUNTIME_SUPPORT_PRINT_H
#define MARCO_RUNTIME_SUPPORT_PRINT_H

#include "marco/Runtime/Support/Mangling.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

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

#endif // MARCO_RUNTIME_SUPPORT_PRINT_H
