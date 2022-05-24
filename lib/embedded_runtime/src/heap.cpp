#include "../include/marco/driver/heap.h"
#include "../include/marco/driver/serial.h"
#include <assert.h>

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
  size_t *header = (size_t*) find_free_space(blk_size);
  if (header) {
  *header = *header | 1;
  } else {
  header = (size_t*) sbrk(blk_size);
  *header = blk_size | 1;
}
  return (char *)header + HEADER_SIZE;
}




void free(void *ptr) {
    size_t *header = (size_t*)((char *)ptr - HEADER_SIZE);
    *header = *header & ~1L; // unmark allocated bit
    ptr=nullptr;
}*/


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

typedef struct Block{
  size_t size;
  struct Block* next;
  bool free;
}block;

#define HEADER_SIZE sizeof(block)

bool first = 1;
block* head_block = NULL;
block* last_allocated = NULL;

block *find_free_space(size_t size){
  block* it;
  it = head_block;
  //while(it->next != nullptr){
  while (it && !(it->free && it->size >= size)){
    serial.write("searching\n\r");
    if(it->free && size <= it->size){
      serial.write("space reused\n\r");
      return it;
    }
    it = it->next;
  }
  return NULL;
}
void *malloc(size_t size){
  size_t blk_size = ALIGN(size + HEADER_SIZE);
  block *header;
  if(size <= 0) return NULL;
  
  header = find_free_space(blk_size);
  header->size = blk_size;
  if(header){
    header->free = 0;
    return (void*) header;
  }else{
    header= (block* ) sbrk(0);
    if(sbrk(blk_size) == (void*)-1){
      return NULL;
      }
    }
    header->free = 0;
    last_allocated->next = header;
    last_allocated = header;
    return (void*) header;
    
  }

void free(void *ptr){
  block *header = (block*) ptr;
  header->free = 0;
  //header->next = nullptr;
  header->size = 0;
  //ptr = nullptr;
}