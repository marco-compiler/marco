#include <stddef.h>

void integerArrayOutputAsArgument(int *array, size_t s1, size_t s2) {
  int value = 1;

  for (size_t i = 0; i < s1; ++i) {
    for (size_t j = 0; j < s2; ++j) {
      array[i * s2 + j] = value++;
    }
  }
}

void realArrayOutputAsArgument(double *array, size_t s1, size_t s2) {
  double value = 1.5;

  for (size_t i = 0; i < s1; ++i) {
    for (size_t j = 0; j < s2; ++j) {
      array[i * s2 + j] = value++;
    }
  }
}