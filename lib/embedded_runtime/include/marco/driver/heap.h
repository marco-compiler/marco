
#define ALIGNMENT 8
#define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT -1))

#define SIZE_T_SIZE (ALIGN(sizeof(st))) // header size
typedef unsigned long st;

void *malloc(st size);

void free(void *ptr);

st sbrk();