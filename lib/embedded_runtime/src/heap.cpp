#include "heap.h"


extern char *_Min_Stack_Size;

st* sbrk(int incr) {
  extern char _end;     /* Defined by the linker */
  static char *heap_end;
  char *prev_heap_end;
 
  if (heap_end == 0) {
    heap_end = &_end;
  }
  prev_heap_end = heap_end;
  if (heap_end + incr > _Min_Stack_Size) {
    return nullptr;
  }

  heap_end += incr;
  return (st*) prev_heap_end;
}


void *malloc(st size) {
    st blk_size = ALIGN(size + SIZE_T_SIZE);
    st *header = sbrk(blk_size);
    *header = blk_size | 1; // mark allocated bit
    return (char *)header + SIZE_T_SIZE;
}

void free(void *ptr) {
    st *header = (st *)ptr - SIZE_T_SIZE;
    *header = *header & ~1L; // unmark allocated bit
}