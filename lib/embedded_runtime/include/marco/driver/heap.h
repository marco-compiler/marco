#ifndef HEAP_H
#define HEAP_H

#include <cstddef>

#define ALIGNMENT 8
#define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT -1))

#define HEADER_SIZE (ALIGN(sizeof(size_t))) // header size -contains packet length and determine whether the block is free or not


void *malloc(size_t size);

void free(void *ptr);

size_t sbrk();

#endif