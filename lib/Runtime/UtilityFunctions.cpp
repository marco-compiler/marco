#include "marco/Runtime/UtilityFunctions.h"
#include "marco/Runtime/Utils.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>

//===----------------------------------------------------------------------===//
// clone
//===----------------------------------------------------------------------===//

namespace
{
  /// Clone an array into another one.
  ///
  /// @tparam T 					destination array type
  /// @tparam U 					source array type
  /// @param destination  destination array
  /// @param values 			source values
  template<typename T, typename U>
  void clone_void(UnrankedMemRefType<T>* destination, UnrankedMemRefType<U>* source)
  {
    DynamicMemRefType dynamicSource(*source);
    DynamicMemRefType dynamicDestination(*destination);

    // Check that the two arrays have the same number of elements
    [[maybe_unused]] int64_t sourceFlatSize = std::accumulate(
        dynamicSource.sizes,
        dynamicSource.sizes + dynamicSource.rank,
        static_cast<int64_t>(1),
        std::multiplies<int64_t>());

    [[maybe_unused]] int64_t destinationFlatSize = std::accumulate(
        dynamicDestination.sizes,
        dynamicDestination.sizes + dynamicDestination.rank,
        static_cast<int64_t>(1),
        std::multiplies<int64_t>());

    assert(sourceFlatSize == destinationFlatSize);

    auto sourceIt = std::begin(dynamicSource);
    auto destinationIt = std::begin(dynamicDestination);

    for (size_t i = 0, e = sourceFlatSize; i < e; ++i) {
      *destinationIt = *sourceIt;

      ++sourceIt;
      ++destinationIt;
    }
  }
}

RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(double))
