#ifndef MARCO_UNITTEST_RUNTIME_UTILS_H
#define MARCO_UNITTEST_RUNTIME_UTILS_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <array>

template<typename T, size_t Size>
StridedMemRefType<T, 1> getMemRef(std::array<T, Size>& data)
{
  StridedMemRefType<T, 1> result;

  result.basePtr = data.data();
  result.data = data.data();
  result.offset = 0;
  result.sizes[0] = Size;
  result.strides[0] = 1;

  return result;
}

template<typename T, size_t Rank>
StridedMemRefType<T, Rank> getMemRef(T* data, std::array<int64_t, Rank> dimensions)
{
  StridedMemRefType<T, Rank> result;

  result.basePtr = data;
  result.data = data;
  result.offset = 0;

  int64_t stride = 1;

  for (size_t i = 0, e = dimensions.size(); i < e; ++i) {
    result.sizes[i] = dimensions[i];
    result.strides[e - i - 1] = stride;
    stride *= dimensions[e - i - 1];
  }

  return result;
}

template<typename T, int Rank>
UnrankedMemRefType<T> getUnrankedMemRef(StridedMemRefType<T, Rank>& memRef)
{
  UnrankedMemRefType<T> result;

  result.rank = Rank;
  result.descriptor = static_cast<void*>(&memRef);

  return result;
}

#endif // MARCO_UNITTEST_RUNTIME_UTILS_H
