#ifndef MARCO_RUNTIME_UTILS_H
#define MARCO_RUNTIME_UTILS_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <iostream>
#include <vector>

template<typename T>
class DynamicMemRefIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

    DynamicMemRefIterator(DynamicMemRefType<T> &descriptor, int64_t offset = 0)
        : offset(offset), descriptor(&descriptor) {
      indices.resize(descriptor.rank, 0);
    }

    DynamicMemRefIterator<T> &operator++() {
      if (descriptor->rank == 0) {
        offset = -1;
        return *this;
      }

      int dim = descriptor->rank - 1;

      while (dim >= 0 && indices[dim] == (descriptor->sizes[dim] - 1)) {
        offset -= indices[dim] * descriptor->strides[dim];
        indices[dim] = 0;
        --dim;
      }

      if (dim < 0) {
        offset = -1;
        return *this;
      }

      ++indices[dim];
      offset += descriptor->strides[dim];
      return *this;
    }

    reference operator*() { return descriptor->data[offset]; }
    pointer operator->() { return &descriptor->data[offset]; }

    const std::vector<int64_t> &getIndices() { return indices; }

    bool operator==(const DynamicMemRefIterator &other) const {
      return other.offset == offset && other.descriptor == descriptor;
    }

    bool operator!=(const DynamicMemRefIterator &other) const {
      return !(*this == other);
    }

  private:
    int64_t offset = 0;
    std::vector<int64_t> indices = {};
    DynamicMemRefType<T>* descriptor;
};

namespace std
{
  template<typename T>
  DynamicMemRefIterator<T> begin(DynamicMemRefType<T>& memRef)
  {
    return DynamicMemRefIterator(memRef);
  }

  template<typename T>
  DynamicMemRefIterator<T> end(DynamicMemRefType<T>& memRef)
  {
    return DynamicMemRefIterator(memRef, -1);
  }
}

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

#endif // MARCO_RUNTIME_UTILS_H
