#ifndef MARCO_RUNTIME_MEMORYMANAGEMENT_H
#define MARCO_RUNTIME_MEMORYMANAGEMENT_H

#include "marco/Runtime/Support/Mangling.h"
#include <cstdint>

void* heapAlloc(int64_t size);

RUNTIME_FUNC_DECL(heapAlloc, PTR(void), int64_t)

extern "C" void* _mlir_memref_to_llvm_alloc(int64_t size);

void heapFree(void* ptr);

RUNTIME_FUNC_DECL(heapFree, void, PTR(void))

extern "C" void _mlir_memref_to_llvm_free(void* ptr);

#endif // MARCO_RUNTIME_MEMORYMANAGEMENT_H
