#ifdef SUNDIALS_ENABLE

#include "marco/Runtime/Solvers/SUNDIALS/Instance.h"
#include <cassert>
#include <iostream>

using namespace ::marco::runtime;
using namespace ::marco::runtime::sundials;

namespace marco::runtime::sundials
{
  VariableDimensions::VariableDimensions(size_t rank)
  {
    dimensions.resize(rank, 0);
  }

  size_t VariableDimensions::rank() const
  {
    return dimensions.size();
  }

  uint64_t& VariableDimensions::operator[](size_t index)
  {
    return dimensions[index];
  }

  const uint64_t& VariableDimensions::operator[](size_t index) const
  {
    return dimensions[index];
  }

  VariableDimensions::const_iterator VariableDimensions::begin() const
  {
    assert(isValid() && "Variable dimensions have not been initialized");
    return dimensions.begin();
  }

  VariableDimensions::const_iterator VariableDimensions::end() const
  {
    assert(isValid() && "Variable dimensions have not been initialized");
    return dimensions.end();
  }

  VariableIndicesIterator VariableDimensions::indicesBegin() const
  {
    assert(isValid() && "Variable dimensions have not been initialized");
    return VariableIndicesIterator::begin(*this);
  }

  VariableIndicesIterator VariableDimensions::indicesEnd() const
  {
    assert(isValid() && "Variable dimensions have not been initialized");
    return VariableIndicesIterator::end(*this);
  }

  bool VariableDimensions::isValid() const
  {
    return std::none_of(
        dimensions.begin(), dimensions.end(), [](const auto& dimension) {
          return dimension == 0;
        });
  }

  VariableIndicesIterator::~VariableIndicesIterator()
  {
    delete[] indices;
  }

  VariableIndicesIterator VariableIndicesIterator::begin(
      const VariableDimensions& dimensions)
  {
    VariableIndicesIterator result(dimensions);

    for (size_t i = 0; i < dimensions.rank(); ++i) {
      result.indices[i] = 0;
    }

    return result;
  }

  VariableIndicesIterator VariableIndicesIterator::end(
      const VariableDimensions& dimensions)
  {
    VariableIndicesIterator result(dimensions);

    for (size_t i = 0; i < dimensions.rank(); ++i) {
      result.indices[i] = dimensions[i];
    }

    return result;
  }

  bool VariableIndicesIterator::operator==(
      const VariableIndicesIterator& it) const
  {
    if (dimensions != it.dimensions) {
      return false;
    }

    for (size_t i = 0; i < dimensions->rank(); ++i) {
      if (indices[i] != it.indices[i]) {
        return false;
      }
    }

    return true;
  }

  bool VariableIndicesIterator::operator!=(
      const VariableIndicesIterator& it) const
  {
    if (dimensions != it.dimensions) {
      return true;
    }

    for (size_t i = 0; i < dimensions->rank(); ++i) {
      if (indices[i] != it.indices[i]) {
        return true;
      }
    }

    return false;
  }

  VariableIndicesIterator& VariableIndicesIterator::operator++()
  {
    fetchNext();
    return *this;
  }

  VariableIndicesIterator VariableIndicesIterator::operator++(int)
  {
    auto temp = *this;
    fetchNext();
    return temp;
  }

  const uint64_t* VariableIndicesIterator::operator*() const
  {
    return indices;
  }

  VariableIndicesIterator::VariableIndicesIterator(
      const VariableDimensions& dimensions)
      : dimensions(&dimensions)
  {
    indices = new uint64_t[dimensions.rank()];
  }

  void VariableIndicesIterator::fetchNext()
  {
    size_t rank = dimensions->rank();
    size_t posFromLast = 0;

    assert(std::none_of(
        dimensions->begin(), dimensions->end(),
        [](const auto& dimension) {
          return dimension == 0;
        }));

    while (posFromLast < rank && ++indices[rank - posFromLast - 1] ==
               (*dimensions)[rank - posFromLast - 1]) {
      ++posFromLast;
    }

    if (posFromLast != rank) {
      for (size_t i = 0; i < posFromLast; ++i) {
        indices[rank - i - 1] = 0;
      }
    }
  }

  bool checkAllocation(void* retval, const char* functionName)
  {
    if (retval == nullptr) {
      std::cerr << "SUNDIALS_ERROR: " << functionName;
      std::cerr << "() failed - returned NULL pointer" << std::endl;
      return false;
    }

    return true;
  }

  void printIndices(const std::vector<int64_t>& indices)
  {
    std::cerr << "[";

    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      if (i != 0) {
        std::cerr << ", ";
      }

      std::cerr << indices[i];
    }

    std::cerr << "]";
  }

  void printIndices(const std::vector<uint64_t>& indices)
  {
    std::cerr << "[";

    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      if (i != 0) {
        std::cerr << ", ";
      }

      std::cerr << indices[i];
    }

    std::cerr << "]";
  }

  bool advanceVariableIndices(
      uint64_t* indices, const VariableDimensions& dimensions)
  {
    for (size_t i = 0, e = dimensions.rank(); i < e; ++i) {
      size_t pos = e - i - 1;
      ++indices[pos];

      if (indices[pos] == dimensions[pos]) {
        indices[pos] = 0;
      } else {
        return true;
      }
    }

    for (size_t i = 0, e = dimensions.rank(); i < e; ++i) {
      indices[i] = dimensions[i];
    }

    return false;
  }

  bool advanceVariableIndices(
      std::vector<uint64_t>& indices, const VariableDimensions& dimensions)
  {
    return advanceVariableIndices(indices.data(), dimensions);
  }

  bool advanceVariableIndicesUntil(
      std::vector<uint64_t>& indices,
      const VariableDimensions& dimensions,
      const std::vector<uint64_t>& end)
  {
    assert(indices.size() == end.size());

    if (!advanceVariableIndices(indices, dimensions)) {
      return false;
    }

    return indices != end;
  }

  bool advanceEquationIndices(
      int64_t* indices, const MultidimensionalRange& ranges)
  {
    for (size_t i = 0, e = ranges.size(); i < e; ++i) {
      size_t pos = e - i - 1;
      ++indices[pos];

      if (indices[pos] == ranges[pos].end) {
        indices[pos] = ranges[pos].begin;
      } else {
        return true;
      }
    }

    for (size_t i = 0, e = ranges.size(); i < e; ++i) {
      indices[i] = ranges[i].end;
    }

    return false;
  }

  bool advanceEquationIndices(
      std::vector<int64_t>& indices, const MultidimensionalRange& ranges)
  {
    return advanceEquationIndices(indices.data(), ranges);
  }

  bool advanceEquationIndicesUntil(
      std::vector<int64_t>& indices,
      const MultidimensionalRange& ranges,
      const std::vector<int64_t>& end)
  {
    assert(indices.size() == end.size());

    if (!advanceEquationIndices(indices, ranges)) {
      return false;
    }

    return indices != end;
  }

  uint64_t getVariableFlatIndex(
      const VariableDimensions& dimensions,
      const uint64_t* indices)
  {
    uint64_t offset = indices[0];

    for (size_t i = 1, e = dimensions.rank(); i < e; ++i) {
      offset = offset * dimensions[i] + indices[i];
    }

    return offset;
  }

  uint64_t getVariableFlatIndex(
      const VariableDimensions& dimensions,
      const std::vector<uint64_t>& indices)
  {
    assert(indices.size() == dimensions.rank());
    return getVariableFlatIndex(dimensions, indices.data());
  }

  uint64_t getEquationFlatIndex(
      const std::vector<int64_t>& indices,
      const MultidimensionalRange& ranges)
  {
    assert(indices[0] >= ranges[0].begin);
    uint64_t offset = indices[0] - ranges[0].begin;

    for (size_t i = 1, e = ranges.size(); i < e; ++i) {
      assert(ranges[i].end > ranges[i].begin);
      offset = offset * (ranges[i].end - ranges[i].begin) +
          (indices[i] - ranges[i].begin);
    }

    assert(offset >= 0);
    return offset;
  }

  void getEquationIndicesFromFlatIndex(
      uint64_t flatIndex,
      std::vector<int64_t>& result,
      const MultidimensionalRange& ranges)
  {
    result.resize(ranges.size());
    uint64_t size = 1;

    for (size_t i = 1, e = ranges.size(); i < e; ++i) {
      assert(ranges[i].end > ranges[i].begin);
      size *= ranges[i].end - ranges[i].begin;
    }

    for (size_t i = 1, e = ranges.size(); i < e; ++i) {
      result[i - 1] =
          static_cast<int64_t>(flatIndex / size) + ranges[i - 1].begin;

      flatIndex %= size;
      assert(ranges[i].end > ranges[i].begin);
      size /= ranges[i].end - ranges[i].begin;
    }

    result[ranges.size() - 1] =
        static_cast<int64_t>(flatIndex) + ranges.back().begin;

    assert(size == 1);

    assert(([&]() -> bool {
             for (size_t i = 0, e = result.size(); i < e; ++i) {
               if (result[i] < ranges[i].begin ||
                   result[i] >= ranges[i].end) {
                 return false;
               }
             }

             return true;
           }()) && "Wrong index unflattening result");
  }
}

#endif // SUNDIALS_ENABLE
