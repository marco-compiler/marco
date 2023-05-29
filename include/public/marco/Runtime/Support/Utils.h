#ifndef MARCO_RUNTIME_SUPPORT_UTILS_H
#define MARCO_RUNTIME_SUPPORT_UTILS_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <iostream>
#include <vector>

namespace impl
{
  template<typename T>
  const T& get(const DynamicMemRefType<T>& descriptor, const std::vector<int64_t>& indices)
  {
    if (descriptor.rank == 0)
      return descriptor.data[descriptor.offset];

    int64_t curOffset = descriptor.offset;

    for (int64_t dim = descriptor.rank - 1; dim >= 0; --dim) {
      int64_t currentIndex = *(indices.begin() + dim);
      assert(currentIndex < descriptor.sizes[dim] && "Index overflow");
      curOffset += currentIndex * descriptor.strides[dim];
    }

    return descriptor.data[curOffset];
  }

  template<typename T>
  void printArrayDescriptor(std::ostream& stream,
                            const DynamicMemRefType<T>& descriptor,
                            std::vector<int64_t>& indexes,
                            int64_t dimension)
  {
    stream << "[";

    for (int64_t i = 0, e = descriptor.sizes[dimension]; i < e; ++i) {
      indexes[dimension] = i;

      if (i > 0) {
        stream << ", ";
      }

      if (dimension == descriptor.rank - 1) {
        stream << get(descriptor, indexes);
      } else {
        printArrayDescriptor(stream, descriptor, indexes, dimension + 1);
      }
    }

    indexes[dimension] = 0;
    stream << "]";
  }
}

template<typename T>
std::ostream& operator<<(
    std::ostream& stream, const DynamicMemRefType<T>& descriptor)
{
  std::vector<int64_t> indexes(descriptor.rank, 0);
  impl::printArrayDescriptor(stream, descriptor, indexes, 0);
  return stream;
}

#endif // MARCO_RUNTIME_SUPPORT_UTILS_H
