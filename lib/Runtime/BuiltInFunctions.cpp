#include "marco/Runtime/BuiltInFunctions.h"
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
  /// @param destination destination matrix
  /// @param values 			source values
  template<typename T, typename U>
  void diagonal_void(UnsizedArrayDescriptor<T> destination, UnsizedArrayDescriptor<U> values)
  {
    // Check that the array is square-like (all the dimensions have the same
    // size). Note that the implementation is generalized to n-D dimensions,
    // while the "identity" Modelica function is defined only for 2-D arrays.
    // Still, the implementation complexity would be the same.

    assert(destination.hasSameSizes());

    // Check that the sizes of the matrix dimensions match with the amount of
    // values to be set.

    assert(destination.getRank() > 0);
    assert(values.getRank() == 1);
    assert(destination.getDimension(0) == values.getDimension(0));

    // Directly use the iterators, as we need to determine the current indexes
    // so that we can place a 1 if the access is on the matrix diagonal.

    for (auto it = destination.begin(), end = destination.end(); it != end; ++it) {
      auto indexes = it.getCurrentIndexes();
      assert(!indexes.empty());

      bool isIdentityAccess = std::all_of(indexes.begin(), indexes.end(), [&indexes](const auto& i) {
        return i == indexes[0];
      });

      *it = isIdentityAccess ? values.get(indexes[0]) : 0;
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
// fill
//===----------------------------------------------------------------------===//

namespace
{
  /// Set all the elements of an array to a given value.
  ///
  /// @tparam T 		 data type
  /// @param array  array to be populated
  /// @param value  value to be set
  template<typename T>
  void fill_void(UnsizedArrayDescriptor<T> array, T value)
  {
    for (auto& element : array) {
      element = value;
    }
  }
}

RUNTIME_FUNC_DEF(fill, void, ARRAY(bool), bool)
RUNTIME_FUNC_DEF(fill, void, ARRAY(int32_t), int32_t)
RUNTIME_FUNC_DEF(fill, void, ARRAY(int64_t), int64_t)
RUNTIME_FUNC_DEF(fill, void, ARRAY(float), float)
RUNTIME_FUNC_DEF(fill, void, ARRAY(double), double)

//===----------------------------------------------------------------------===//
// identity
//===----------------------------------------------------------------------===//

namespace
{
  /// Set a multi-dimensional array to an identity like matrix.
  ///
  /// @tparam T 	   data type
  /// @param array  array to be populated
  template<typename T>
  void identity_void(UnsizedArrayDescriptor<T> array)
  {
    // Check that the array is square-like (all the dimensions have the same
    // size). Note that the implementation is generalized to n-D dimensions,
    // while the "identity" Modelica function is defined only for 2-D arrays.
    // Still, the implementation complexity would be the same.

    assert(array.hasSameSizes());

    // Directly use the iterators, as we need to determine the current indexes
    // so that we can place a 1 if the access is on the matrix diagonal.

    for (auto it = array.begin(), end = array.end(); it != end; ++it) {
      auto indexes = it.getCurrentIndexes();
      assert(!indexes.empty());

      bool isIdentityAccess = std::all_of(indexes.begin(), indexes.end(), [&indexes](const auto& i) {
        return i == indexes[0];
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
// linspace
//===----------------------------------------------------------------------===//

namespace
{
  /// Populate a 1-D array with equally spaced elements.
  ///
  /// @tparam T 		 data type
  /// @param array  array to be populated
  /// @param start  start value
  /// @param end 	 end value
  template<typename T>
  void linspace_void(UnsizedArrayDescriptor<T> array, double start, double end)
  {
    using dimension_t = typename UnsizedArrayDescriptor<T>::dimension_t;
    assert(array.getRank() == 1);

    auto n = array.getDimension(0);
    double step = (end - start) / ((double) n - 1);

    for (dimension_t i = 0; i < n; ++i) {
      array.get(i) = start + static_cast<double>(i) * step;
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
    assert(value > 0);
    return std::log(value);
  }

  double log_f64(double value)
  {
    assert(value > 0);
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
    assert(value > 0);
    return std::log10(value);
  }

  double log10_f64(double value)
  {
    assert(value > 0);
    return std::log10(value);
  }
}

RUNTIME_FUNC_DEF(log10, float, float)
RUNTIME_FUNC_DEF(log10, double, double)

//===----------------------------------------------------------------------===//
// max
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  T max(UnsizedArrayDescriptor<T> array)
  {
    return *std::max_element(array.begin(), array.end());
  }

  bool max_i1(UnsizedArrayDescriptor<bool> array)
  {
    return std::any_of(array.begin(), array.end(), [](const bool& value) {
      return value;
    });
  }

  int32_t max_i32(UnsizedArrayDescriptor<int32_t> array)
  {
    return ::max(array);
  }

  int32_t max_i64(UnsizedArrayDescriptor<int64_t> array)
  {
    return ::max(array);
  }

  float max_f32(UnsizedArrayDescriptor<float> array)
  {
    return ::max(array);
  }

  double max_f64(UnsizedArrayDescriptor<double> array)
  {
    return ::max(array);
  }
}

RUNTIME_FUNC_DEF(max, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(max, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(max, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(max, float, ARRAY(float))
RUNTIME_FUNC_DEF(max, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// max
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  T max(T x, T y)
  {
    return std::max(x, y);
  }

  bool max_i1(bool x, bool y)
  {
    return x || y;
  }

  int32_t max_i32(int32_t x, int32_t y)
  {
    return max(x, y);
  }

  int64_t max_i64(int64_t x, int64_t y)
  {
    return ::max(x, y);
  }

  float max_f32(float x, float y)
  {
    return ::max(x, y);
  }

  double max_f64(double x, double y)
  {
    return ::max(x, y);
  }
}

RUNTIME_FUNC_DEF(max, bool, bool, bool)
RUNTIME_FUNC_DEF(max, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(max, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(max, float, float, float)
RUNTIME_FUNC_DEF(max, double, double, double)

//===----------------------------------------------------------------------===//
// min
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  T min(UnsizedArrayDescriptor<T> array)
  {
    return *std::min_element(array.begin(), array.end());
  }

  bool min_i1(UnsizedArrayDescriptor<bool> array)
  {
    return std::all_of(array.begin(), array.end(), [](const bool& value) {
      return value;
    });
  }

  int32_t min_i32(UnsizedArrayDescriptor<int32_t> array)
  {
    return ::min(array);
  }

  int32_t min_i64(UnsizedArrayDescriptor<int64_t> array)
  {
    return ::min(array);
  }

  float min_f32(UnsizedArrayDescriptor<float> array)
  {
    return ::min(array);
  }

  double min_f64(UnsizedArrayDescriptor<double> array)
  {
    return ::min(array);
  }
}

RUNTIME_FUNC_DEF(min, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(min, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(min, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(min, float, ARRAY(float))
RUNTIME_FUNC_DEF(min, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// min
//===----------------------------------------------------------------------===//

namespace
{
  template<typename T>
  T min(T x, T y)
  {
    return std::min(x, y);
  }

  bool min_i1(bool x, bool y)
  {
    return x && y;
  }

  int32_t min_i32(int32_t x, int32_t y)
  {
    return ::min(x, y);
  }

  int64_t min_i64(int64_t x, int64_t y)
  {
    return ::min(x, y);
  }

  float min_f32(float x, float y)
  {
    return ::min(x, y);
  }

  double min_f64(double x, double y)
  {
    return ::min(x, y);
  }
}

RUNTIME_FUNC_DEF(min, bool, bool, bool)
RUNTIME_FUNC_DEF(min, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(min, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(min, float, float, float)
RUNTIME_FUNC_DEF(min, double, double, double)

//===----------------------------------------------------------------------===//
// ones
//===----------------------------------------------------------------------===//

namespace
{
  /// Set all the elements of an array to ones.
  ///
  /// @tparam T data type
  /// @param array   array to be populated
  template<typename T>
  void ones_void(UnsizedArrayDescriptor<T> array)
  {
    for (auto& element : array) {
      element = 1;
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
  T product(UnsizedArrayDescriptor<T> array)
  {
    return std::accumulate(array.begin(), array.end(), 1, std::multiplies<T>());
  }

  bool product_i1(UnsizedArrayDescriptor<bool> array)
  {
    return std::all_of(array.begin(), array.end(), [](const bool& value) {
      return value;
    });
  }

  int32_t product_i32(UnsizedArrayDescriptor<int32_t> array)
  {
    return ::product(array);
  }

  int64_t product_i64(UnsizedArrayDescriptor<int64_t> array)
  {
    return ::product(array);
  }

  float product_f32(UnsizedArrayDescriptor<float> array)
  {
    return ::product(array);
  }

  double product_f64(UnsizedArrayDescriptor<double> array)
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
  T sum(UnsizedArrayDescriptor<T> array)
  {
    return std::accumulate(array.begin(), array.end(), 0, std::plus<T>());
  }

  bool sum_i1(UnsizedArrayDescriptor<bool> array)
  {
    return std::any_of(array.begin(), array.end(), [](const bool& value) {
      return value;
    });
  }

  int32_t sum_i32(UnsizedArrayDescriptor<int32_t> array)
  {
    return ::sum(array);
  }

  int64_t sum_i64(UnsizedArrayDescriptor<int64_t> array)
  {
    return ::sum(array);
  }

  float sum_f32(UnsizedArrayDescriptor<float> array)
  {
    return ::sum(array);
  }

  double sum_f64(UnsizedArrayDescriptor<double> array)
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
  /// @tparam Destination	destination type
  /// @tparam Source				source type
  /// @param destination		destination matrix
  /// @param source				source matrix
  template<typename Destination, typename Source>
  void symmetric_void(UnsizedArrayDescriptor<Destination> destination, UnsizedArrayDescriptor<Source> source)
  {
    using dimension_t = typename UnsizedArrayDescriptor<Destination>::dimension_t;

    // The two arrays must have exactly two dimensions
    assert(destination.getRank() == 2);
    assert(source.getRank() == 2);

    // The two matrixes must have the same dimensions
    assert(destination.getDimension(0) == source.getDimension(0));
    assert(destination.getDimension(1) == source.getDimension(1));

    auto size = destination.getDimension(0);

    // Manually iterate on the dimensions, so that we can explore just half
    // of the source matrix.

    for (dimension_t i = 0; i < size; ++i) {
      for (dimension_t j = i; j < size; ++j) {
        destination.set({ i, j }, source.get({ i, j }));

        if (i != j) {
          destination.set({j, i}, source.get({ i, j }));
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
  /// @tparam Source 		 source type
  /// @param destination  destination matrix
  /// @param source  		 source matrix
  template<typename Destination, typename Source>
  void transpose_void(UnsizedArrayDescriptor<Destination> destination, UnsizedArrayDescriptor<Source> source)
  {
    using dimension_t = typename UnsizedArrayDescriptor<Source>::dimension_t;

    // The two arrays must have exactly two dimensions
    assert(destination.getRank() == 2);
    assert(source.getRank() == 2);

    // The two matrixes must have transposed dimensions
    assert(destination.getDimension(0) == source.getDimension(1));
    assert(destination.getDimension(1) == source.getDimension(0));

    // Directly use the iterators, as we need to determine the current
    // indexes and transpose them to access the other matrix.

    for (auto it = source.begin(), end = source.end(); it != end; ++it) {
      auto indexes = it.getCurrentIndexes();
      assert(indexes.size() == 2);

      std::vector<dimension_t> transposedIndexes;

      for (auto revIt = indexes.rbegin(), revEnd = indexes.rend(); revIt != revEnd; ++revIt) {
        transposedIndexes.push_back(*revIt);
      }

      destination.set(transposedIndexes, *it);
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
  void zeros_void(UnsizedArrayDescriptor<T> array)
  {
    for (auto& element : array) {
      element = 0;
    }
  }
}

RUNTIME_FUNC_DEF(zeros, void, ARRAY(bool))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(float))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(double))
