#include "../include/marco/driver/heap.h"
#include "../include/marco/driver/serial.h"

extern SerialPort serial;
/*

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




void free(void *ptr) {
    size_t *header = (size_t*)((char *)ptr - HEADER_SIZE);
    *header = *header & ~1L; // unmark allocated bit
    ptr=nullptr;
}
*/
/*
static uint8_t *__sbrk_heap_end = NULL;
extern uint8_t _end;
extern uint8_t _stack_top; 
extern uint32_t _Min_Stack_Size; 
const uint32_t stack_limit = (uint32_t)&_stack_top - (uint32_t)&_Min_Stack_Size;
const uint8_t *max_heap = (uint8_t *) stack_limit;
uint8_t *prev_heap_end;
*/

static size_t *__sbrk_heap_end = NULL;
extern size_t _end;
extern size_t _stack_top; 
extern size_t _Min_Stack_Size; 
const size_t stack_limit = (size_t)&_stack_top - (size_t)&_Min_Stack_Size;
const size_t *max_heap = (size_t *) stack_limit;
size_t *prev_heap_end;

void *sbrk(ptrdiff_t incr)
{
  if (NULL == __sbrk_heap_end)
  {
    __sbrk_heap_end = &_end;
  }

  if (__sbrk_heap_end + incr > max_heap)
  {
    serial.write("\r\n Out Of Memory \r\n");
    return (void *)-1;
  }

  prev_heap_end = __sbrk_heap_end;
  __sbrk_heap_end += incr;
  return (void *)prev_heap_end;
  
}
size_t *find_free_space(size_t size) {
  size_t *header = (size_t*)_end;
  while (header < __sbrk_heap_end) {
  if (!(*header & 1) && *header >= size){
    serial.write("\r\n free space found \r\n");
    return header;
  }
    
    header = (size_t*)((void*)header + (*header & ~1L));
  }
  return nullptr;
}
void *malloc(size_t size){
  size_t blk_size = ALIGN(size + HEADER_SIZE);
  size_t *header = find_free_space(blk_size);
  if(header){
    *header = *header | 1;
  }else{
    header = (size_t* ) sbrk(0);
    if(sbrk(blk_size) == (void*)-1){
      return NULL;
      }
    }
    //serial.write((int)*header);
    //sserial.write("\n\r");
    return (void*) header;
  }

/*
void *malloc(size_t size){
  size_t blk_size = ALIGN(size + HEADER_SIZE);
  void *p;
  p = sbrk(0);
  if(sbrk(blk_size) == (void*)-1)
    return NULL;
  else
  return p;
}
*/
void free(void *ptr) {
    size_t *header;
    header = (size_t *)(ptr - HEADER_SIZE);
    *header = *header & ~1L;
    header = NULL;
    ptr = (void*)NULL;
}