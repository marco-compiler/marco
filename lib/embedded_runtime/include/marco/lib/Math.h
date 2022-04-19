#ifndef MARCO_RUNTIME_MATH_H
#define MARCO_RUNTIME_MATH_H

#include "marco/runtime/Mangling.h"
#include <cstdint>

// Pow with 'bool' return type
RUNTIME_FUNC_DECL(pow, bool, bool, bool)
RUNTIME_FUNC_DECL(pow, bool, bool, int32_t)
RUNTIME_FUNC_DECL(pow, bool, bool, int64_t)
RUNTIME_FUNC_DECL(pow, bool, bool, float)
RUNTIME_FUNC_DECL(pow, bool, bool, double)

RUNTIME_FUNC_DECL(pow, bool, int32_t, bool)
RUNTIME_FUNC_DECL(pow, bool, int32_t, int32_t)
RUNTIME_FUNC_DECL(pow, bool, int32_t, float)

RUNTIME_FUNC_DECL(pow, bool, int64_t, bool)
RUNTIME_FUNC_DECL(pow, bool, int64_t, int64_t)
RUNTIME_FUNC_DECL(pow, bool, int64_t, double)

RUNTIME_FUNC_DECL(pow, bool, float, bool)
RUNTIME_FUNC_DECL(pow, bool, float, int32_t)
RUNTIME_FUNC_DECL(pow, bool, float, float)

RUNTIME_FUNC_DECL(pow, bool, double, bool)
RUNTIME_FUNC_DECL(pow, bool, double, int64_t)
RUNTIME_FUNC_DECL(pow, bool, double, double)

// Pow with 'int32_t' return type
RUNTIME_FUNC_DECL(pow, int32_t, bool, bool)
RUNTIME_FUNC_DECL(pow, int32_t, bool, int32_t)
RUNTIME_FUNC_DECL(pow, int32_t, bool, float)

RUNTIME_FUNC_DECL(pow, int32_t, int32_t, bool)
RUNTIME_FUNC_DECL(pow, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DECL(pow, int32_t, int32_t, float)

RUNTIME_FUNC_DECL(pow, int32_t, float, bool)
RUNTIME_FUNC_DECL(pow, int32_t, float, int32_t)
RUNTIME_FUNC_DECL(pow, int32_t, float, float)

// Pow with 'int64_t' return type
RUNTIME_FUNC_DECL(pow, int64_t, bool, bool)
RUNTIME_FUNC_DECL(pow, int64_t, bool, int64_t)
RUNTIME_FUNC_DECL(pow, int64_t, bool, double)

RUNTIME_FUNC_DECL(pow, int64_t, int64_t, bool)
RUNTIME_FUNC_DECL(pow, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DECL(pow, int64_t, int64_t, double)

RUNTIME_FUNC_DECL(pow, int64_t, double, bool)
RUNTIME_FUNC_DECL(pow, int64_t, double, int64_t)
RUNTIME_FUNC_DECL(pow, int64_t, double, double)

// Pow with 'float' return type
RUNTIME_FUNC_DECL(pow, float, bool, bool)
RUNTIME_FUNC_DECL(pow, float, bool, int32_t)
RUNTIME_FUNC_DECL(pow, float, bool, float)

RUNTIME_FUNC_DECL(pow, float, int32_t, bool)
RUNTIME_FUNC_DECL(pow, float, int32_t, int32_t)
RUNTIME_FUNC_DECL(pow, float, int32_t, float)

RUNTIME_FUNC_DECL(pow, float, float, bool)
RUNTIME_FUNC_DECL(pow, float, float, int32_t)
RUNTIME_FUNC_DECL(pow, float, float, float)

// Pow with 'double' return type
RUNTIME_FUNC_DECL(pow, double, bool, bool)
RUNTIME_FUNC_DECL(pow, double, bool, int64_t)
RUNTIME_FUNC_DECL(pow, double, bool, double)

RUNTIME_FUNC_DECL(pow, double, int32_t, bool)
RUNTIME_FUNC_DECL(pow, double, int32_t, int64_t)
RUNTIME_FUNC_DECL(pow, double, int32_t, double)

RUNTIME_FUNC_DECL(pow, double, int64_t, bool)
RUNTIME_FUNC_DECL(pow, double, int64_t, int64_t)
RUNTIME_FUNC_DECL(pow, double, int64_t, double)

RUNTIME_FUNC_DECL(pow, double, double, bool)
RUNTIME_FUNC_DECL(pow, double, double, int64_t)
RUNTIME_FUNC_DECL(pow, double, double, double)

#endif // MARCO_RUNTIME_MATH_H
