#ifndef HEAP_H
#define HEAP_H

#include <stdint.h>
#include <cstddef>

#define ALIGNMENT 8
#define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT -1))

#define HEADER_SIZE (ALIGN(sizeof(size_t))) // header size -contains packet length and determine whether the block is free or not


extern "C" void *malloc(size_t size);

extern "C" void free(void *ptr);

size_t sbrk();

//extern "C" void *sbrk(ptrdiff_t incr);

#endif