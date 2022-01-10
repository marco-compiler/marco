#include "heap.h"


extern char *_Min_Stack_Size;
extern char _end;     
size_t heap_last;


size_t* sbrk(const int incr) {
 
  static char *heap_end;
  char *prev_heap_end;
  extern char _end;

  if (heap_end == 0) {
    heap_end = &_end;
  }
  prev_heap_end = heap_end;
  if (heap_end + incr > _Min_Stack_Size) {
    return nullptr;
  }

  heap_end += incr;
  heap_last = (size_t) heap_end;
  return (size_t*) prev_heap_end;
}


size_t *find_free_space(size_t size) {
  size_t *header = (size_t*)_end;
  while (*header < heap_last) {
  if (!(*header & 1) && *header >= size)
    return header;
    header = (size_t*)((char*)header + (*header & ~1L));
  }
  return nullptr;
}


void *malloc(size_t size) {
  size_t blk_size = ALIGN(size + HEADER_SIZE);
  size_t *header = find_free_space(blk_size);
  if (header) {
  *header = *header | 1;
  } else {
  header = sbrk(blk_size);
  *header = blk_size | 1;
}
  return (char *)header + HEADER_SIZE;
}

/*
void *malloc(size_t size) {
size_t blk_size = ALIGN(size + HEADER_SIZE);
size_t *header = sbrk(blk_size);
*header = blk_size | 1; // mark allocated bit
return (char *)header + HEADER_SIZE;
}
*/


void free(void *ptr) {
    size_t *header = (size_t*)((char *)ptr - HEADER_SIZE);
    *header = *header & ~1L; // unmark allocated bit
    ptr=nullptr;
}


