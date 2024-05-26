#ifndef MARCO_RUNTIME_MEMORYMANAGEMENT_H
#define MARCO_RUNTIME_MEMORYMANAGEMENT_H

#include <cstdint>

extern "C"
{
  void* marco_malloc(int64_t size);
  void* marco_realloc(void* ptr, int64_t size);
  void marco_free(void* ptr);
};

#endif // MARCO_RUNTIME_MEMORYMANAGEMENT_H
