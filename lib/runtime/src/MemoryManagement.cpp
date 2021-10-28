#include <marco/runtime/MemoryManagement.h>

inline void* heapAlloc(int64_t sizeInBytes)
{
	return malloc(sizeInBytes);
}

RUNTIME_FUNC_DEF(heapAlloc, PTR(void), int64_t)

inline void heapFree(void* ptr)
{
	free(ptr);
}

RUNTIME_FUNC_DEF(heapFree, void, PTR(void))
