#ifndef MARCO_RUNTIME_MEMORYMANAGEMENT_H
#define MARCO_RUNTIME_MEMORYMANAGEMENT_H

#include <cstdint>

#include "Mangling.h"

RUNTIME_FUNC_DECL(heapAlloc, PTR(void), int64_t)
RUNTIME_FUNC_DECL(heapFree, void, PTR(void))

#endif	// MARCO_RUNTIME_MEMORYMANAGEMENT_H
