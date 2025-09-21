#include <stddef.h>

int sumIntegerArray(int *array, size_t s1, size_t s2) {
  int result = 0;

  for (size_t i = 0; i < s1; ++i) {
    for (size_t j = 0; j < s2; ++j) {
      result += array[i * s2 + j];
    }
  }

  return result;
}

double sumRealArray(double *array, size_t s1, size_t s2) {
  double result = 0;

  for (size_t i = 0; i < s1; ++i) {
    for (size_t j = 0; j < s2; ++j) {
      result += array[i * s2 + j];
    }
  }

  return result;
}
