#ifndef MARCO_RUNTIME_MEMORYMANAGEMENT_H
#define MARCO_RUNTIME_MEMORYMANAGEMENT_H

#include "marco/Runtime/Mangling.h"
#include <cstdint>

RUNTIME_FUNC_DECL(heapAlloc, PTR(void), int64_t)
RUNTIME_FUNC_DECL(heapFree, void, PTR(void))

void* heapAlloc(int64_t size);
void heapFree(void* ptr);

#endif	// MARCO_RUNTIME_MEMORYMANAGEMENT_H
