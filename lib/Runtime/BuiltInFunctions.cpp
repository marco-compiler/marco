#include "marco/Runtime/BuiltInFunctions.h"
#include "marco/Runtime/Utils.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

//===----------------------------------------------------------------------===//
// abs
//===----------------------------------------------------------------------===//

namespace
{
  bool abs_i1(bool value)
  {
    return value;
  }

  int32_t abs_i32(int32_t value)
  {
    return std::abs(value);
  }

  int64_t abs_i64(int64_t value)
  {
    return std::abs(value);
  }

  float abs_f32(float value)
  {
    return std::abs(value);
  }

  double abs_f64(double value)
  {
    return std::abs(value);
  }
}

RUNTIME_FUNC_DEF(abs, bool, bool)
RUNTIME_FUNC_DEF(abs, int32_t, int32_t)
RUNTIME_FUNC_DEF(abs, int64_t, int64_t)
RUNTIME_FUNC_DEF(abs, float, float)
RUNTIME_FUNC_DEF(abs, double, double)

//===----------------------------------------------------------------------===//
// acos
//===----------------------------------------------------------------------===//

namespace
{
  float acos_f32(float value)
  {
    return std::acos(value);
  }

  double acos_f64(double value)
  {
    return std::acos(value);
  }
}

RUNTIME_FUNC_DEF(acos, float, float)
RUNTIME_FUNC_DEF(acos, double, double)

//===----------------------------------------------------------------------===//
// asin
//===----------------------------------------------------------------------===//

namespace
{
  float asin_f32(float value)
  {
    return std::asin(value);
  }

  double asin_f64(double value)
  {
    return std::asin(value);
  }
}

RUNTIME_FUNC_DEF(asin, float, float)
RUNTIME_FUNC_DEF(asin, double, double)

//===----------------------------------------------------------------------===//
// atan
//===----------------------------------------------------------------------===//

namespace
{
  float atan_f32(float value)
  {
    return std::atan(value);
  }

  double atan_f64(double value)
  {
    return std::atan(value);
  }
}

RUNTIME_FUNC_DEF(atan, float, float)
RUNTIME_FUNC_DEF(atan, double, double)

//===----------------------------------------------------------------------===//
// atan2
//===----------------------------------------------------------------------===//

namespace
{
  float atan2_f32(float y, float x)
  {
    return std::atan2(y, x);
  }

  double atan2_f64(double y, double x)
  {
    return std::atan2(y, x);
  }
}

RUNTIME_FUNC_DEF(atan2, float, float, float)
RUNTIME_FUNC_DEF(atan2, double, double, double)

//===----------------------------------------------------------------------===//
// ceil
//===----------------------------------------------------------------------===//

namespace
{
  bool ceil_i1(bool value)
  {
    return value;
  }

  int32_t ceil_i32(int32_t value)
  {
    return value;
  }

  int64_t ceil_i64(int64_t value)
  {
    return value;
  }

  float ceil_f32(float value)
  {
    return std::ceil(value);
  }

  double ceil_f64(double value)
  {
    return std::ceil(value);
  }
}

RUNTIME_FUNC_DEF(ceil, bool, bool)
RUNTIME_FUNC_DEF(ceil, int32_t, int32_t)
RUNTIME_FUNC_DEF(ceil, int64_t, int64_t)
RUNTIME_FUNC_DEF(ceil, float, float)
RUNTIME_FUNC_DEF(ceil, double, double)

//===----------------------------------------------------------------------===//
// cos
//===----------------------------------------------------------------------===//

namespace
{
  float cos_f32(float value)
  {
    return std::cos(value);
  }

  double cos_f64(double value)
  {
    return std::cos(value);
  }
}

RUNTIME_FUNC_DEF(cos, float, float)
RUNTIME_FUNC_DEF(cos, double, double)

//===----------------------------------------------------------------------===//
// cosh
//===----------------------------------------------------------------------===//

namespace
{
  float cosh_f32(float value)
  {
    return std::cosh(value);
  }

  double cosh_f64(double value)
  {
    return std::cosh(value);
  }
}

RUNTIME_FUNC_DEF(cosh, float, float)
RUNTIME_FUNC_DEF(cosh, double, double)

//===----------------------------------------------------------------------===//
// diagonal
//===----------------------------------------------------------------------===//

namespace
{
  /// Place some values on the diagonal of a matrix, and set all the other
  /// elements to zero.
  ///
  /// @tparam T 					destination matrix type
  /// @tparam U 					source values type
  /// @param destination  destination matrix
  /// @param values 			source values
  template<typename T, typename U>
  void diagonal_void(UnrankedMemRefType<T>* destination, UnrankedMemRefType<U>* values)
  {
    DynamicMemRefType dynamicDestination(*destination);

    // Check that the array is square-like (all the dimensions have the same
    // size). Note that the implementation is generalized to n-D dimensions,
    // while the "identity" Modelica function is defined only for 2-D arrays.
    // Still, the implementation complexity would be the same.

    assert(std::all_of(dynamicDestination.sizes, dynamicDestination.sizes + dynamicDestination.rank, [&](int64_t dimension) {
      return dimension == dynamicDestination.sizes[0];
    }));

    // Check that the sizes of the matrix dimensions match with the amount of
    // values to be set.
    assert(dynamicDestination.rank > 0);

    assert(values->rank == 1);
    auto valuesDesc = static_cast<StridedMemRefType<U, 1>*>(values->descriptor);

    assert(dynamicDestination.sizes[0] == valuesDesc->sizes[0]);

    // Directly use the iterators, as we need to determine the current indexes
    // so that we can place a 1 if the access is on the matrix diagonal.

    for (auto it = std::begin(dynamicDestination), end = std::end(dynamicDestination); it != end; ++it) {
      auto indices = it.getIndices();
      assert(!indices.empty());

      bool isIdentityAccess = std::all_of(indices.begin(), indices.end(), [&indices](const auto& i) {
        return i == indices[0];
      });

      *it = isIdentityAccess ? (*valuesDesc)[indices[0]] : 0;
    }
  }
}

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(double))

//===----------------------------------------------------------------------===//
// div
//===----------------------------------------------------------------------===//

namespace
{
  bool div_i1(bool x, bool y)
  {
    assert(y && "Division by zero");
    return x;

    return x;
  }

  int32_t div_i32(int32_t x, int32_t y)
  {
    return x / y;
  }

  int64_t div_i64(int64_t x, int64_t y)
  {
    return x / y;
  }

  float div_f32(float x, float y)
  {
    return std::trunc(x / y);
  }

  double div_f64(double x, double y)
  {
    return std::trunc(x / y);
  }
}

RUNTIME_FUNC_DEF(div, bool, bool, bool)
RUNTIME_FUNC_DEF(div, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(div, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(div, float, float, float)
RUNTIME_FUNC_DEF(div, double, double, double)

//===----------------------------------------------------------------------===//
// exp
//===----------------------------------------------------------------------===//

namespace
{
  float exp_f32(float value)
  {
    return std::exp(value);
  }

  double exp_f64(double value)
  {
    return std::exp(value);
  }
}

RUNTIME_FUNC_DEF(exp, float, float)
RUNTIME_FUNC_DEF(exp, double, double)

//===----------------------------------------------------------------------===//
// floor
//===----------------------------------------------------------------------===//

namespace
{
  bool floor_i1(bool value)
  {
    return value;
  }

  int32_t floor_i32(int32_t value)
  {
    return value;
  }

  int64_t floor_i64(int64_t value)
  {
    return value;
  }

  float floor_f32(float value)
  {
    return std::floor(value);
  }

  double floor_f64(double value)
  {
    return std::floor(value);
  }
}

RUNTIME_FUNC_DEF(floor, bool, bool)
RUNTIME_FUNC_DEF(floor, int32_t, int32_t)
RUNTIME_FUNC_DEF(floor, int64_t, int64_t)
RUNTIME_FUNC_DEF(floor, float, float)
RUNTIME_FUNC_DEF(floor, double, double)

//===----------------------------------------------------------------------===//
// identity
//===----------------------------------------------------------------------===//

namespace
{
  /// Set a multi-dimensional array to an identity like matrix.
  ///
  /// @tparam T 	    data type
  /// @param array    array to be populated
  template<typename T>
  void identity_void(UnrankedMemRefType<T>* array)
  {
    DynamicMemRefType dynamicArray(*array);

    // Check that the array is square-like (all the dimensions have the same
    // size). Note that the implementation is generalized to n-D dimensions,
    // while the "identity" Modelica function is defined only for 2-D arrays.
    // Still, the implementation complexity would be the same.

    assert(std::all_of(dynamicArray.sizes, dynamicArray.sizes + dynamicArray.rank, [&](int64_t dimension) {
      return dimension == dynamicArray.sizes[0];
    }));

    // Directly use the iterators, as we need to determine the current indexes
    // so that we can place a 1 if the access is on the matrix diagonal.

    for (auto it = std::begin(dynamicArray), end = std::end(dynamicArray); it != end; ++it) {
      const auto& indices = it.getIndices();
      assert(!indices.empty());

      bool isIdentityAccess = std::all_of(indices.begin(), indices.end(), [&indices](const auto& i) {
        return i == indices[0];
      });

      *it = isIdentityAccess ? 1 : 0;
    }
  }
}

RUNTIME_FUNC_DEF(identity, void, ARRAY(bool))
RUNTIME_FUNC_DEF(identity, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(identity, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(identity, void, ARRAY(float))
RUNTIME_FUNC_DEF(identity, void, ARRAY(double))

//===----------------------------------------------------------------------===//
// integer
//===----------------------------------------------------------------------===//

namespace
{
  bool integer_i1(bool value)
  {
    return value;
  }

  int32_t integer_i32(int32_t value)
  {
    return value;
  }

  int64_t integer_i64(int64_t value)
  {
    return value;
  }

  float integer_f32(float value)
  {
    return std::floor(value);
  }

  double integer_f64(double value)
  {
    return std::floor(value);
  }
}

RUNTIME_FUNC_DEF(integer, bool, bool)
RUNTIME_FUNC_DEF(integer, int32_t, int32_t)
RUNTIME_FUNC_DEF(integer, int64_t, int64_t)
RUNTIME_FUNC_DEF(integer, float, float)
RUNTIME_FUNC_DEF(integer, double, double)

//===----------------------------------------------------------------------===//
// linspace
//===----------------------------------------------------------------------===//

namespace
{
  /// Populate a 1-D array with equally spaced elements.
  ///
  /// @tparam T 		 data type
  /// @param array  array to be populated
  /// @param start  start value
  /// @param end 	    end value
  template<typename T>
  void linspace_void(UnrankedMemRefType<T>* array, double start, double end)
  {
    assert(array->rank == 1);
    auto arrayDesc = static_cast<StridedMemRefType<T, 1>*>(array->descriptor);

    auto n = arrayDesc->sizes[0];
    double step = (end - start) / ((double) n - 1);

    for (int64_t i = 0; i < n; ++i) {
      (*arrayDesc)[i] = start + static_cast<double>(i) * step;
    }
  }
}

RUNTIME_FUNC_DEF(linspace, void, ARRAY(bool), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(bool), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int32_t), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int32_t), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int64_t), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int64_t), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(float), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(float), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(double), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(double), double, double)

//===----------------------------------------------------------------------===//
// ln
//===----------------------------------------------------------------------===//

namespace
{
  float log_f32(float value)
  {
    assert(value >= 0);
    return std::log(value);
  }

  double log_f64(double value)
  {
    assert(value >= 0);
    return std::log(value);
  }
}

RUNTIME_FUNC_DEF(log, float, float)
RUNTIME_FUNC_DEF(log, double, double)

//===----------------------------------------------------------------------===//
// log
//===----------------------------------------------------------------------===//

namespace
{
  float log10_f32(float value)
  {
    assert(value >= 0);
    return std::log10(value);
  }

  double log10_f64(double value)
  {
    assert(value >= 0);
    return std::log10(value);
  }
}

RUNTIME_FUNC_DEF(log10, float, float)
RUNTIME_FUNC_DEF(log10, double, double)

//===----------------------------------------------------------------------===//
// maxArray
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  T maxArray(UnrankedMemRefType<T>* array)
  {
    DynamicMemRefType dynamicArray(*array);
    return *std::max_element(std::begin(dynamicArray), std::end(dynamicArray));
  }

  bool maxArray_i1(UnrankedMemRefType<bool>* array)
  {
    DynamicMemRefType dynamicArray(*array);

    return std::any_of(std::begin(dynamicArray), std::end(dynamicArray), [](const bool& value) {
      return value;
    });
  }

  int32_t maxArray_i32(UnrankedMemRefType<int32_t>* array)
  {
    return ::maxArray(array );
  }

  int32_t maxArray_i64(UnrankedMemRefType<int64_t>* array)
  {
    return ::maxArray(array);
  }

  float maxArray_f32(UnrankedMemRefType<float>* array)
  {
    return ::maxArray(array);
  }

  double maxArray_f64(UnrankedMemRefType<double>* array)
  {
    return ::maxArray(array);
  }
}

RUNTIME_FUNC_DEF(maxArray, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(maxArray, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(maxArray, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(maxArray, float, ARRAY(float))
RUNTIME_FUNC_DEF(maxArray, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// maxScalars
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  T maxScalars(T x, T y)
  {
    return std::max(x, y);
  }

  bool maxScalars_i1(bool x, bool y)
  {
    return x || y;
  }

  int32_t maxScalars_i32(int32_t x, int32_t y)
  {
    return ::maxScalars(x, y);
  }

  int64_t maxScalars_i64(int64_t x, int64_t y)
  {
    return ::maxScalars(x, y);
  }

  float maxScalars_f32(float x, float y)
  {
    return ::maxScalars(x, y);
  }

  double maxScalars_f64(double x, double y)
  {
    return ::maxScalars(x, y);
  }
}

RUNTIME_FUNC_DEF(maxScalars, bool, bool, bool)
RUNTIME_FUNC_DEF(maxScalars, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(maxScalars, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(maxScalars, float, float, float)
RUNTIME_FUNC_DEF(maxScalars, double, double, double)

//===----------------------------------------------------------------------===//
// minArray
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  T minArray(UnrankedMemRefType<T>* array)
  {
    DynamicMemRefType dynamicArray(*array);
    return *std::min_element(std::begin(dynamicArray), std::end(dynamicArray));
  }

  bool minArray_i1(UnrankedMemRefType<bool>* array)
  {
    DynamicMemRefType dynamicArray(*array);

    return std::all_of(std::begin(dynamicArray), std::end(dynamicArray), [](const bool& value) {
      return value;
    });
  }

  int32_t minArray_i32(UnrankedMemRefType<int32_t>* array)
  {
    return ::minArray(array);
  }

  int32_t minArray_i64(UnrankedMemRefType<int64_t>* array)
  {
    return ::minArray(array);
  }

  float minArray_f32(UnrankedMemRefType<float>* array)
  {
    return ::minArray(array);
  }

  double minArray_f64(UnrankedMemRefType<double>* array)
  {
    return ::minArray(array);
  }
}

RUNTIME_FUNC_DEF(minArray, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(minArray, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(minArray, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(minArray, float, ARRAY(float))
RUNTIME_FUNC_DEF(minArray, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// minScalars
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  T minScalars(T x, T y)
  {
    return std::min(x, y);
  }

  bool minScalars_i1(bool x, bool y)
  {
    return x && y;
  }

  int32_t minScalars_i32(int32_t x, int32_t y)
  {
    return ::minScalars(x, y);
  }

  int64_t minScalars_i64(int64_t x, int64_t y)
  {
    return ::minScalars(x, y);
  }

  float minScalars_f32(float x, float y)
  {
    return ::minScalars(x, y);
  }

  double minScalars_f64(double x, double y)
  {
    return ::minScalars(x, y);
  }
}

RUNTIME_FUNC_DEF(minScalars, bool, bool, bool)
RUNTIME_FUNC_DEF(minScalars, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(minScalars, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(minScalars, float, float, float)
RUNTIME_FUNC_DEF(minScalars, double, double, double)

//===----------------------------------------------------------------------===//
// mod
//===----------------------------------------------------------------------===//

namespace
{
  bool mod_i1(bool x, bool y)
  {
    assert(y && "Division by zero");

    if (y) {
      return false;
    }

    return x;
  }

  int32_t mod_i32(int32_t x, int32_t y)
  {
    return x - std::floor(static_cast<float>(x) / y) * y;
  }

  int64_t mod_i64(int64_t x, int64_t y)
  {
    return x - std::floor(static_cast<double>(x) / y) * y;
  }

  float mod_f32(float x, float y)
  {
    return x - std::floor(x / y) * y;
  }

  double mod_f64(double x, double y)
  {
    return x - std::floor(x / y) * y;
  }
}

RUNTIME_FUNC_DEF(mod, bool, bool, bool)
RUNTIME_FUNC_DEF(mod, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(mod, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(mod, float, float, float)
RUNTIME_FUNC_DEF(mod, double, double, double)

//===----------------------------------------------------------------------===//
// ones
//===----------------------------------------------------------------------===//

namespace
{
  /// Set all the elements of an array to zero.
  ///
  /// @tparam T data type
  /// @param array   array to be populated
  template<typename T>
  void ones_void(UnrankedMemRefType<T>* array)
  {
    DynamicMemRefType dynamicArray(*array);

    for (auto it = std::begin(dynamicArray), end = std::end(dynamicArray); it != end; ++it) {
      *it = 1;
    }
  }
}

RUNTIME_FUNC_DEF(ones, void, ARRAY(bool))
RUNTIME_FUNC_DEF(ones, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(ones, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(ones, void, ARRAY(float))
RUNTIME_FUNC_DEF(ones, void, ARRAY(double))

//===----------------------------------------------------------------------===//
// product
//===----------------------------------------------------------------------===//

namespace
{
  /// Multiply all the elements of an array.
  ///
  /// @tparam T 		 data type
  /// @param array  array
  /// @return product of all the values
  template<typename T>
  T product(UnrankedMemRefType<T>* array)
  {
    DynamicMemRefType dynamicArray(*array);
    return std::accumulate(std::begin(dynamicArray), std::end(dynamicArray), static_cast<T>(1), std::multiplies<T>());
  }

  bool product_i1(UnrankedMemRefType<bool>* array)
  {
    DynamicMemRefType dynamicArray(*array);

    return std::all_of(std::begin(dynamicArray), std::end(dynamicArray), [](const bool& value) {
      return value;
    });
  }

  int32_t product_i32(UnrankedMemRefType<int32_t>* array)
  {
    return ::product(array);
  }

  int64_t product_i64(UnrankedMemRefType<int64_t>* array)
  {
    return ::product(array);
  }

  float product_f32(UnrankedMemRefType<float>* array)
  {
    return ::product(array);
  }

  double product_f64(UnrankedMemRefType<double>* array)
  {
    return ::product(array);
  }
}

RUNTIME_FUNC_DEF(product, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(product, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(product, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(product, float, ARRAY(float))
RUNTIME_FUNC_DEF(product, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// rem
//===----------------------------------------------------------------------===//

namespace
{
  bool rem_i1(bool x, bool y)
  {
    assert(y && "Division by zero");

    if (y) {
      return false;
    }

    return x;
  }

  int32_t rem_i32(int32_t x, int32_t y)
  {
    return x % y;
  }

  int64_t rem_i64(int64_t x, int64_t y)
  {
    return x % y;
  }

  float rem_f32(float x, float y)
  {
    return std::fmod(x, y);
  }

  double rem_f64(double x, double y)
  {
    return std::fmod(x, y);
  }
}

RUNTIME_FUNC_DEF(rem, bool, bool, bool)
RUNTIME_FUNC_DEF(rem, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(rem, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(rem, float, float, float)
RUNTIME_FUNC_DEF(rem, double, double, double)

//===----------------------------------------------------------------------===//
// sign
//===----------------------------------------------------------------------===//

namespace
{
  /// Get the sign of a value.
  ///
  /// @tparam T		 data type
  /// @param value  value
  /// @return 1 if value is > 0, -1 if < 0, 0 if = 0
  template<typename T>
  long sign(T value)
  {
    if (value == 0) {
      return 0;
    }

    if (value > 0) {
      return 1;
    }

    return -1;
  }

  template<typename T>
  int32_t sign_i32(T value)
  {
    return ::sign(value);
  }

  template<>
  int32_t sign_i32(bool value)
  {
    return value ? 1 : 0;
  }

  template<typename T>
  int64_t sign_i64(T value)
  {
    return ::sign(value);
  }

  template<>
  int64_t sign_i64(bool value)
  {
    return value ? 1 : 0;
  }
}

RUNTIME_FUNC_DEF(sign, int32_t, bool)
RUNTIME_FUNC_DEF(sign, int32_t, int32_t)
RUNTIME_FUNC_DEF(sign, int32_t, int64_t)
RUNTIME_FUNC_DEF(sign, int32_t, float)
RUNTIME_FUNC_DEF(sign, int32_t, double)

RUNTIME_FUNC_DEF(sign, int64_t, bool)
RUNTIME_FUNC_DEF(sign, int64_t, int32_t)
RUNTIME_FUNC_DEF(sign, int64_t, int64_t)
RUNTIME_FUNC_DEF(sign, int64_t, float)
RUNTIME_FUNC_DEF(sign, int64_t, double)

//===----------------------------------------------------------------------===//
// sin
//===----------------------------------------------------------------------===//

namespace
{
  float sin_f32(float value)
  {
    return std::sin(value);
  }

  double sin_f64(double value)
  {
    return std::sin(value);
  }
}

RUNTIME_FUNC_DEF(sin, float, float)
RUNTIME_FUNC_DEF(sin, double, double)

//===----------------------------------------------------------------------===//
// sinh
//===----------------------------------------------------------------------===//

namespace
{
  float sinh_f32(float value)
  {
    return std::sinh(value);
  }

  double sinh_f64(double value)
  {
    return std::sinh(value);
  }
}

RUNTIME_FUNC_DEF(sinh, float, float)
RUNTIME_FUNC_DEF(sinh, double, double)

//===----------------------------------------------------------------------===//
// sqrt
//===----------------------------------------------------------------------===//

namespace
{
  float sqrt_f32(float value)
  {
    assert(value >= 0);
    return std::sqrt(value);
  }

  double sqrt_f64(double value)
  {
    assert(value >= 0);
    return std::sqrt(value);
  }
}

RUNTIME_FUNC_DEF(sqrt, float, float)
RUNTIME_FUNC_DEF(sqrt, double, double)

//===----------------------------------------------------------------------===//
// sum
//===----------------------------------------------------------------------===//

namespace
{
  /// Sum all the elements of an array.
  ///
  /// @tparam T 		 data type
  /// @param array  array
  /// @return sum of all the values
  template<typename T>
  T sum(UnrankedMemRefType<T>* array)
  {
    DynamicMemRefType dynamicArray(*array);
    return std::accumulate(std::begin(dynamicArray), std::end(dynamicArray), static_cast<T>(0), std::plus<T>());
  }

  bool sum_i1(UnrankedMemRefType<bool>* array)
  {
    DynamicMemRefType dynamicArray(*array);

    return std::any_of(std::begin(dynamicArray), std::end(dynamicArray), [](const bool& value) {
      return value;
    });
  }

  int32_t sum_i32(UnrankedMemRefType<int32_t>* array)
  {
    return ::sum(array);
  }

  int64_t sum_i64(UnrankedMemRefType<int64_t>* array)
  {
    return ::sum(array);
  }

  float sum_f32(UnrankedMemRefType<float>* array)
  {
    return ::sum(array);
  }

  double sum_f64(UnrankedMemRefType<double>* array)
  {
    return ::sum(array);
  }
}

RUNTIME_FUNC_DEF(sum, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(sum, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(sum, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(sum, float, ARRAY(float))
RUNTIME_FUNC_DEF(sum, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// symmetric
//===----------------------------------------------------------------------===//

namespace
{
  /// Populate the destination matrix so that it becomes the symmetric version
  /// of the source one, thus discarding the elements below the source diagonal.
  ///
  /// @tparam Destination  destination type
  /// @tparam Source			 source type
  /// @param destination	 destination matrix
  /// @param source				 source matrix
  template<typename Destination, typename Source>
  void symmetric_void(UnrankedMemRefType<Destination>* destination, UnrankedMemRefType<Source>* source)
  {
    // The two arrays must have exactly two dimensions
    assert(destination->rank == 2);
    assert(source->rank == 2);

    auto destinationMatrix = static_cast<StridedMemRefType<Destination, 2>*>(destination->descriptor);
    auto sourceMatrix = static_cast<StridedMemRefType<Source, 2>*>(source->descriptor);

    // The two matrices must have the same dimensions
    assert(destinationMatrix->sizes[0] == sourceMatrix->sizes[0]);
    assert(destinationMatrix->sizes[1] == sourceMatrix->sizes[1]);

    auto size = destinationMatrix->sizes[0];

    // Manually iterate on the dimensions, so that we can explore just half
    // of the source matrix.

    std::array<int64_t, 2> indices;
    std::array<int64_t, 2> transposedIndices;

    for (int64_t i = 0; i < size; ++i) {
      indices[0] = i;
      transposedIndices[1] = i;

      for (int64_t j = i; j < size; ++j) {
        indices[1] = j;
        transposedIndices[0] = j;

        (*destinationMatrix)[indices] = (*sourceMatrix)[indices];

        if (i != j) {
          (*destinationMatrix)[transposedIndices] = (*sourceMatrix)[indices];
        }
      }
    }
  }
}

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(double))

//===----------------------------------------------------------------------===//
// tan
//===----------------------------------------------------------------------===//

namespace
{
  float tan_f32(float value)
  {
    return std::tan(value);
  }

  double tan_f64(double value)
  {
    return std::tan(value);
  }
}

RUNTIME_FUNC_DEF(tan, float, float)
RUNTIME_FUNC_DEF(tan, double, double)

//===----------------------------------------------------------------------===//
// tanh
//===----------------------------------------------------------------------===//

namespace
{
  float tanh_f32(float value)
  {
    return std::tanh(value);
  }

  double tanh_f64(double value)
  {
    return std::tanh(value);
  }
}

RUNTIME_FUNC_DEF(tanh, float, float)
RUNTIME_FUNC_DEF(tanh, double, double)

//===----------------------------------------------------------------------===//
// transpose
//===----------------------------------------------------------------------===//

namespace
{
  /// Transpose a matrix.
  ///
  /// @tparam Destination destination type
  /// @tparam Source 		  source type
  /// @param destination  destination matrix
  /// @param source  		  source matrix
  template<typename Destination, typename Source>
  void transpose_void(UnrankedMemRefType<Destination>* destination, UnrankedMemRefType<Source>* source)
  {
    // The two arrays must have exactly two dimensions
    assert(destination->rank == 2);
    assert(source->rank == 2);

    auto destinationMatrix = static_cast<StridedMemRefType<Destination, 2>*>(destination->descriptor);
    auto sourceMatrix = static_cast<StridedMemRefType<Source, 2>*>(source->descriptor);

    // The two matrices must have transposed dimensions
    assert(destinationMatrix->sizes[0] == sourceMatrix->sizes[1]);
    assert(destinationMatrix->sizes[1] == sourceMatrix->sizes[0]);

    // Directly use the iterators, as we need to determine the current
    // indexes and transpose them to access the other matrix.

    for (auto it = sourceMatrix->begin(), end = sourceMatrix->end(); it != end; ++it) {
      const auto& indexes = it.getIndices();
      assert(indexes.size() == 2);

      std::vector<int64_t> transposedIndexes;

      for (auto revIt = indexes.rbegin(), revEnd = indexes.rend(); revIt != revEnd; ++revIt) {
        transposedIndexes.push_back(*revIt);
      }

      (*destinationMatrix)[transposedIndexes] = *it;
    }
  }
}

RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(double))

//===----------------------------------------------------------------------===//
// zeros
//===----------------------------------------------------------------------===//

namespace
{
  /// Set all the elements of an array to zero.
  ///
  /// @tparam T data type
  /// @param array   array to be populated
  template<typename T>
  void zeros_void(UnrankedMemRefType<T>* array)
  {
    DynamicMemRefType dynamicArray(*array);

    for (auto it = std::begin(dynamicArray), end = std::end(dynamicArray); it != end; ++it) {
      *it = 0;
    }
  }
}

RUNTIME_FUNC_DEF(zeros, void, ARRAY(bool))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(float))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(double))
