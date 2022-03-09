#ifndef MARCO_RUNTIME_UTILITYFUNCTIONS_H
#define MARCO_RUNTIME_UTILITYFUNCTIONS_H

#include "marco/runtime/ArrayDescriptor.h"
#include "marco/runtime/Mangling.h"
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

#endif	// MARCO_RUNTIME_UTILITYFUNCTIONS_H
