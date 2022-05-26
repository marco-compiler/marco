#include "../include/marco/driver/heap.h"
#include "../include/marco/driver/serial.h"
#include <assert.h>

extern SerialPort serial;


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

block* head_block = NULL;
block* last_allocated = NULL;

block *find_free_space(size_t size){
  block* it;
  it = head_block;
  while(it != nullptr){
    /*
    serial.write("searching\n\r");
    serial.write((int)(it->free) );
    serial.write(" Size ");
    serial.write((int)(it->size ));
    serial.write(" Block size ");
    serial.write((int)size);
    */
    if(it->free && size <= it->size){
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
  
  if(header){
    header->free = 0;
    header->next = NULL;
    header->size = blk_size;
    return (header+1);
  }else{
    header= (block* ) sbrk(0);
    if(sbrk(blk_size) == (void*)-1){
      return NULL;
      }
    }
    header->free = 0;
    header->next = NULL;
    header->size = blk_size;
    if(!head_block) 
      head_block = header;
    else last_allocated->next = header;
    last_allocated = header;
    return (header+1);
    
  }

void free(void *ptr){
  block *header ;
  header = (block*) ptr - 1;
  header->free = 1;
}