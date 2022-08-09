#ifndef MARCO_RUNTIME_UTILITYFUNCTIONS_H
#define MARCO_RUNTIME_UTILITYFUNCTIONS_H

#include "marco/Runtime/Mangling.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>

RUNTIME_FUNC_DECL(clone, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DECL(clone, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DECL(clone, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DECL(clone, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DECL(clone, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DECL(clone, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DECL(clone, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DECL(clone, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DECL(clone, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DECL(clone, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DECL(clone, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DECL(clone, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DECL(clone, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DECL(clone, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DECL(clone, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DECL(clone, void, ARRAY(double), ARRAY(double))

#endif	// MARCO_RUNTIME_UTILITYFUNCTIONS_H
