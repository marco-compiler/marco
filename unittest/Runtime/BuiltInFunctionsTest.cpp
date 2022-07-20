#include "marco/Runtime/BuiltInFunctions.h"
#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include <cmath>
#include <numeric>

template<typename T, unsigned int N> using ArraySizes =
		std::array<typename ArrayDescriptor<T, N>::dimension_t, N>;

TEST(Runtime, abs_i1)
{
  auto absFn = [](bool value) -> bool {
    return NAME_MANGLED(abs, bool, bool)(value);
  };

  EXPECT_EQ(absFn(false), false);
  EXPECT_EQ(absFn(true), true);
}

TEST(Runtime, abs_i32)
{
  auto absFn = [](int32_t value) -> int32_t {
    return NAME_MANGLED(abs, int32_t, int32_t)(value);
  };

  EXPECT_EQ(absFn(-5), 5);
  EXPECT_EQ(absFn(0), 0);
  EXPECT_EQ(absFn(5), 5);
}

TEST(Runtime, abs_i64)
{
  auto absFn = [](int64_t value) -> int64_t {
    return NAME_MANGLED(abs, int64_t, int64_t)(value);
  };

  EXPECT_EQ(absFn(-5), 5);
  EXPECT_EQ(absFn(0), 0);
  EXPECT_EQ(absFn(5), 5);
}

TEST(Runtime, abs_f32)
{
  auto absFn = [](float value) -> float {
    return NAME_MANGLED(abs, float, float)(value);
  };

  EXPECT_FLOAT_EQ(absFn(-5), 5);
  EXPECT_FLOAT_EQ(absFn(0), 0);
  EXPECT_FLOAT_EQ(absFn(5), 5);
}

TEST(Runtime, abs_f64)
{
  auto absFn = [](double value) -> double {
    return NAME_MANGLED(abs, double, double)(value);
  };

  EXPECT_DOUBLE_EQ(absFn(-5), 5);
  EXPECT_DOUBLE_EQ(absFn(0), 0);
  EXPECT_DOUBLE_EQ(absFn(5), 5);
}

TEST(Runtime, acos_f32)
{
  auto acosFn = [](float value) -> float {
    return NAME_MANGLED(acos, float, float)(value);
  };

	EXPECT_NEAR(acosFn(1), 0, 0.000001);
	EXPECT_NEAR(acosFn(0.866025403403), M_PI / 6, 0.000001);
	EXPECT_NEAR(acosFn(0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(acosFn(0), M_PI / 2, 0.000001);
	EXPECT_NEAR(acosFn(-0.707106781), M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(acosFn(-0.866025403), M_PI * 5 / 6, 0.000001);
	EXPECT_NEAR(acosFn(-1), M_PI, 0.000001);
}

TEST(Runtime, acos_f64)
{
  auto acosFn = [](double value) -> double {
    return NAME_MANGLED(acos, double, double)(value);
  };

	EXPECT_NEAR(acosFn(1), 0, 0.00001);
	EXPECT_NEAR(acosFn(0.866025403), M_PI / 6, 0.000001);
	EXPECT_NEAR(acosFn(0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(acosFn(0), M_PI / 2, 0.000001);
	EXPECT_NEAR(acosFn(-0.707106781), M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(acosFn(-0.866025403), M_PI * 5 / 6, 0.000001);
	EXPECT_NEAR(acosFn(-1), M_PI, 0.000001);
}

TEST(Runtime, asin_f32)
{
  auto asinFn = [](float value) -> float {
    return NAME_MANGLED(asin, float, float)(value);
  };

	EXPECT_NEAR(asinFn(1), M_PI / 2, 0.000001);
	EXPECT_NEAR(asinFn(0.866025403), M_PI / 3, 0.000001);
	EXPECT_NEAR(asinFn(0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(asinFn(0), 0, 0.000001);
	EXPECT_NEAR(asinFn(-0.707106781), -1 * M_PI / 4, 0.000001);
	EXPECT_NEAR(asinFn(-0.866025403), -1 * M_PI / 3, 0.000001);
	EXPECT_NEAR(asinFn(-1), -1 * M_PI / 2, 0.000001);
}

TEST(Runtime, asin_f64)
{
  auto asinFn = [](double value) -> double {
    return NAME_MANGLED(asin, double, double)(value);
  };

	EXPECT_NEAR(asinFn(1), M_PI / 2, 0.000001);
	EXPECT_NEAR(asinFn(0.866025403), M_PI / 3, 0.000001);
	EXPECT_NEAR(asinFn(0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(asinFn(0), 0, 0.000001);
	EXPECT_NEAR(asinFn(-0.707106781), -1 * M_PI / 4, 0.000001);
	EXPECT_NEAR(asinFn(-0.866025403), -1 * M_PI / 3, 0.000001);
	EXPECT_NEAR(asinFn(-1), -1 * M_PI / 2, 0.000001);
}

TEST(Runtime, atan_f32)
{
  auto atanFn = [](float value) -> float {
    return NAME_MANGLED(atan, float, float)(value);
  };

	EXPECT_NEAR(atanFn(1), M_PI / 4, 0.000001);
	EXPECT_NEAR(atanFn(0.577350269), M_PI / 6, 0.000001);
	EXPECT_NEAR(atanFn(0), 0, 0.000001);
	EXPECT_NEAR(atanFn(-0.577350269), -1 * M_PI / 6, 0.000001);
	EXPECT_NEAR(atanFn(-1), -1 * M_PI / 4, 0.000001);
}

TEST(Runtime, atan_f64)
{
  auto atanFn = [](double value) -> double {
    return NAME_MANGLED(atan, double, double)(value);
  };

	EXPECT_NEAR(atanFn(1), M_PI / 4, 0.000001);
	EXPECT_NEAR(atanFn(0.577350269), M_PI / 6, 0.000001);
	EXPECT_NEAR(atanFn(0), 0, 0.0000001);
	EXPECT_NEAR(atanFn(-0.577350269), -1 * M_PI / 6, 0.000001);
	EXPECT_NEAR(atanFn(-1), -1 * M_PI / 4, 0.000001);
}

TEST(Runtime, atan2_f32)
{
  auto atan2Fn = [](float y, float x) -> float {
    return NAME_MANGLED(atan2, float, float, float)(y, x);
  };

	EXPECT_NEAR(atan2Fn(0.707106781, 0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(atan2Fn(0.707106781, -0.707106781), M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(atan2Fn(-0.707106781, -0.707106781), -1 * M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(atan2Fn(-0.707106781, 0.707106781), -1 * M_PI / 4, 0.000001);
}

TEST(Runtime, atan2_f64)
{
  auto atan2Fn = [](double y, double x) -> double {
    return NAME_MANGLED(atan2, double, double, double)(y, x);
  };

	EXPECT_NEAR(atan2Fn(0.707106781, 0.707106781), M_PI / 4, 0.000001);
	EXPECT_NEAR(atan2Fn(0.707106781, -0.707106781), M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(atan2Fn(-0.707106781, -0.707106781), -1 * M_PI * 3 / 4, 0.000001);
	EXPECT_NEAR(atan2Fn(-0.707106781, 0.707106781), -1 * M_PI / 4, 0.000001);
}


TEST(Runtime, ceil_i1)	 // NOLINT
{
  auto ceilFn = [](bool value) -> bool {
    return NAME_MANGLED(ceil, bool, bool)(value);
  };

  EXPECT_EQ(ceilFn(false), false);
  EXPECT_EQ(ceilFn(true), true);
}

TEST(Runtime, ceil_i32)	 // NOLINT
{
  auto ceilFn = [](int32_t value) -> int32_t {
    return NAME_MANGLED(ceil, int32_t, int32_t)(value);
  };

  EXPECT_EQ(ceilFn(-3), -3);
  EXPECT_EQ(ceilFn(3), 3);
}

TEST(Runtime, ceil_i64)	 // NOLINT
{
  auto ceilFn = [](int64_t value) -> int64_t {
    return NAME_MANGLED(ceil, int64_t, int64_t)(value);
  };

  EXPECT_EQ(ceilFn(-3), -3);
  EXPECT_EQ(ceilFn(3), 3);
}

TEST(Runtime, ceil_f32)	 // NOLINT
{
  auto ceilFn = [](float value) -> float {
    return NAME_MANGLED(ceil, float, float)(value);
  };

  EXPECT_NEAR(ceilFn(-3.14), -3, 0.000001);
  EXPECT_NEAR(ceilFn(3.14), 4, 0.000001);
}

TEST(Runtime, ceil_f64)	 // NOLINT
{
  auto ceilFn = [](double value) -> double {
    return NAME_MANGLED(ceil, double, double)(value);
  };

  EXPECT_NEAR(ceilFn(-3.14), -3, 0.000001);
  EXPECT_NEAR(ceilFn(3.14), 4, 0.000001);
}

TEST(Runtime, cos_f32)
{
  auto cosFn = [](float value) -> float {
    return NAME_MANGLED(cos, float, float)(value);
  };

  EXPECT_NEAR(cosFn(0), 1, 0.000001);
	EXPECT_NEAR(cosFn(M_PI / 6), 0.866025403, 0.000001);
	EXPECT_NEAR(cosFn(M_PI / 4), 0.707106781, 0.000001);
	EXPECT_NEAR(cosFn(M_PI / 2), 0, 0.000001);
	EXPECT_NEAR(cosFn(M_PI), -1, 0.000001);
	EXPECT_NEAR(cosFn(2 * M_PI), 1, 0.000001);
}

TEST(Runtime, cos_f64)
{
  auto cosFn = [](double value) -> double {
    return NAME_MANGLED(cos, double, double)(value);
  };

	EXPECT_NEAR(cosFn(0), 1, 0.000001);
	EXPECT_NEAR(cosFn(M_PI / 6), 0.866025403, 0.000001);
	EXPECT_NEAR(cosFn(M_PI / 4), 0.707106781, 0.000001);
	EXPECT_NEAR(cosFn(M_PI / 2), 0, 0.000001);
	EXPECT_NEAR(cosFn(M_PI), -1, 0.000001);
	EXPECT_NEAR(cosFn(2 * M_PI), 1, 0.000001);
}

TEST(Runtime, cosh_f32)
{
  auto coshFn = [](float value) -> float {
    return NAME_MANGLED(cosh, float, float)(value);
  };

	EXPECT_NEAR(coshFn(0), 1, 0.000001);
	EXPECT_NEAR(coshFn(1), 1.543080634, 0.000001);
}

TEST(Runtime, cosh_f64)
{
  auto coshFn = [](double value) -> double {
    return NAME_MANGLED(cosh, double, double)(value);
  };

	EXPECT_NEAR(coshFn(0), 1, 0.000001);
	EXPECT_NEAR(coshFn(1), 1.543080634, 0.000001);
}

TEST(Runtime, diagonalSquareMatrix_i1_i1)
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(bool))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : false);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i1_i32)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 2, 2, 2 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(int32_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i1_i64)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 2, 2, 2 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(int64_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i1_f32)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 2, 2, 2 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(float))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i1_f64)	 // NOLINT
{
	std::array<bool, 9> destination = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 2, 2, 2 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(double))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] > 0 : false);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i32_i1)	 // NOLINT
{
	std::array<int32_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(bool))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i32_i32)	 // NOLINT
{
	std::array<int, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(int32_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i32_i64)	 // NOLINT
{
	std::array<int32_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(int64_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i32_f32)	 // NOLINT
{
	std::array<int32_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(float))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i32_f64)	 // NOLINT
{
	std::array<int32_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(double))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i64_i1)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(bool))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i64_i32)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(int32_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i64_i64)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(int64_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i64_f32)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(float))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_i64_f64)	 // NOLINT
{
	std::array<int64_t, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(double))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f32_i1)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(bool))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f32_i32)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(int32_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f32_i64)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(int64_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f32_f32)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(float))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f32_f64)	 // NOLINT
{
	std::array<float, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(double))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_FLOAT_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f64_i1)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<bool, 3> values = { true, true, true };
	ArrayDescriptor<bool, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<bool> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(bool))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? (values[i] ? 1 : 0) : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f64_i32)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<int32_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int32_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(int32_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f64_i64)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<int64_t, 3> values = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<int64_t> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(int64_t))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f64_f32)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<float, 3> values = { 1, 2, 3 };
	ArrayDescriptor<float, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<float> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(float))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, diagonalSquareMatrix_f64_f64)	 // NOLINT
{
	std::array<double, 9> destination = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> destinationSizes = { 3, 3 };

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), destinationSizes);
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	std::array<double, 3> values = { 1, 2, 3 };
	ArrayDescriptor<double, 1> valuesDescriptor(values);
	UnsizedArrayDescriptor<double> unsizedValues(valuesDescriptor);

	NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(double))(&unsizedDestination, &unsizedValues);

	for (size_t i = 0; i < destinationSizes[0]; ++i) {
    for (size_t j = 0; j < destinationSizes[1]; j++) {
      EXPECT_DOUBLE_EQ(destination[3 * i + j], i == j ? values[i] : 0);
    }
  }
}

TEST(Runtime, exp_f32)	 // NOLINT
{
  auto expFn = [](float exponent) -> float {
    return NAME_MANGLED(exp, float, float)(exponent);
  };

	EXPECT_NEAR(expFn(0), 1, 0.000001);
	EXPECT_NEAR(expFn(1), 2.718281, 0.000001);
	EXPECT_NEAR(expFn(2), 7.389056, 0.000001);
	EXPECT_NEAR(expFn(-2), 0.135335, 0.000001);
}

TEST(Runtime, exp_f64)	 // NOLINT
{
  auto expFn = [](double exponent) -> double {
    return NAME_MANGLED(exp, double, double)(exponent);
  };

	EXPECT_NEAR(expFn(0), 1, 0.000001);
	EXPECT_NEAR(expFn(1), 2.718281, 0.000001);
	EXPECT_NEAR(expFn(2), 7.389056, 0.000001);
	EXPECT_NEAR(expFn(-2), 0.135335, 0.000001);
}

TEST(Runtime, fill_i1)	 // NOLINT
{
	std::array<bool, 3> data = { false, false, false };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	bool value = true;
	NAME_MANGLED(fill, void, ARRAY(bool), bool)(&unsized, value);

  EXPECT_TRUE(llvm::all_of(data, [&](const auto& element) {
    return element == value;
  }));
}

TEST(Runtime, fill_i32)	 // NOLINT
{
	std::array<int32_t, 3> data = { 0, 0, 0 };
	ArrayDescriptor<int32_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	int value = 1;
	NAME_MANGLED(fill, void, ARRAY(int32_t), int32_t)(&unsized, value);

  EXPECT_TRUE(llvm::all_of(data, [&](const auto& element) {
    return element == value;
  }));
}

TEST(Runtime, fill_i64)	 // NOLINT
{
	std::array<int64_t, 3> data = { 0, 0, 0 };
	ArrayDescriptor<int64_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	long value = 1;
	NAME_MANGLED(fill, void, ARRAY(int64_t), int64_t)(&unsized, value);

  EXPECT_TRUE(llvm::all_of(data, [&](const auto& element) {
    return element == value;
  }));
}

TEST(Runtime, fill_f32)	 // NOLINT
{
	std::array<float, 3> data = { 0, 0, 0 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	float value = 1;
	NAME_MANGLED(fill, void, ARRAY(float), float)(&unsized, value);

  EXPECT_TRUE(llvm::all_of(data, [&](const auto& element) {
    return element == value;
  }));
}

TEST(Runtime, fill_f64)	 // NOLINT
{
	std::array<double, 3> data = { 0, 0, 0 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	double value = 1;
	NAME_MANGLED(fill, void, ARRAY(double), double)(&unsized, value);

  EXPECT_TRUE(llvm::all_of(data, [&](const auto& element) {
    return element == value;
  }));
}

TEST(Runtime, floor_i1)	 // NOLINT
{
  auto floorFn = [](bool value) -> bool {
    return NAME_MANGLED(floor, bool, bool)(value);
  };

  EXPECT_EQ(floorFn(false), false);
  EXPECT_EQ(floorFn(true), true);
}

TEST(Runtime, floor_i32)	 // NOLINT
{
  auto floorFn = [](int32_t value) -> int32_t {
    return NAME_MANGLED(floor, int32_t, int32_t)(value);
  };

  EXPECT_EQ(floorFn(-3), -3);
  EXPECT_EQ(floorFn(3), 3);
}

TEST(Runtime, floor_i64)	 // NOLINT
{
  auto floorFn = [](int64_t value) -> int64_t {
    return NAME_MANGLED(floor, int64_t, int64_t)(value);
  };

  EXPECT_EQ(floorFn(-3), -3);
  EXPECT_EQ(floorFn(3), 3);
}

TEST(Runtime, floor_f32)	 // NOLINT
{
  auto floorFn = [](float value) -> float {
    return NAME_MANGLED(floor, float, float)(value);
  };

  EXPECT_NEAR(floorFn(-3.14), -4, 0.000001);
  EXPECT_NEAR(floorFn(3.14), 3, 0.000001);
}

TEST(Runtime, floor_f64)	 // NOLINT
{
  auto floorFn = [](double value) -> double {
    return NAME_MANGLED(floor, double, double)(value);
  };

  EXPECT_NEAR(floorFn(-3.14), -4, 0.000001);
  EXPECT_NEAR(floorFn(3.14), 3, 0.000001);
}

TEST(Runtime, identitySquareMatrix_i1)	 // NOLINT
{
	std::array<bool, 9> data = { false, true, true, true, false, true, true, true, false };
	ArraySizes<bool, 2> sizes = { 3, 3 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(bool))(&unsized);

	for (size_t i = 0; i < sizes[0]; ++i) {
    for (size_t j = 0; j < sizes[1]; j++) {
      EXPECT_EQ(data[3 * i + j], i == j);
    }
  }
}

TEST(Runtime, identitySquareMatrix_i32)	 // NOLINT
{
	std::array<int32_t, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int32_t, 2> sizes = { 3, 3 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(int32_t))(&unsized);

	for (size_t i = 0; i < sizes[0]; ++i) {
    for (size_t j = 0; j < sizes[1]; j++) {
      EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
    }
  }
}

TEST(Runtime, identitySquareMatrix_i64)	 // NOLINT
{
	std::array<int64_t, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<int64_t, 2> sizes = { 3, 3 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(int64_t))(&unsized);

	for (size_t i = 0; i < sizes[0]; ++i) {
    for (size_t j = 0; j < sizes[1]; j++) {
      EXPECT_EQ(data[3 * i + j], i == j ? 1 : 0);
    }
  }
}

TEST(Runtime, identitySquareMatrix_f32)	 // NOLINT
{
	std::array<float, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<float, 2> sizes = { 3, 3 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(float))(&unsized);

	for (size_t i = 0; i < sizes[0]; ++i) {
    for (size_t j = 0; j < sizes[1]; j++) {
      EXPECT_FLOAT_EQ(data[3 * i + j], i == j ? 1 : 0);
    }
  }
}

TEST(Runtime, identitySquareMatrix_f64)	 // NOLINT
{
	std::array<double, 9> data = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
	ArraySizes<double, 2> sizes = { 3, 3 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	NAME_MANGLED(identity, void, ARRAY(double))(&unsized);

	for (size_t i = 0; i < sizes[0]; ++i) {
    for (size_t j = 0; j < sizes[1]; j++) {
      EXPECT_DOUBLE_EQ(data[3 * i + j], i == j ? 1 : 0);
    }
  }
}

TEST(Runtime, integer_i1)	 // NOLINT
{
  auto integerFn = [](bool value) -> bool {
    return NAME_MANGLED(integer, bool, bool)(value);
  };

  EXPECT_EQ(integerFn(false), false);
  EXPECT_EQ(integerFn(true), true);
}

TEST(Runtime, integer_i32)	 // NOLINT
{
  auto integerFn = [](int32_t value) -> int32_t {
    return NAME_MANGLED(integer, int32_t, int32_t)(value);
  };

  EXPECT_EQ(integerFn(-3), -3);
  EXPECT_EQ(integerFn(3), 3);
}

TEST(Runtime, integer_i64)	 // NOLINT
{
  auto integerFn = [](int64_t value) -> int64_t {
    return NAME_MANGLED(integer, int64_t, int64_t)(value);
  };

  EXPECT_EQ(integerFn(-3), -3);
  EXPECT_EQ(integerFn(3), 3);
}

TEST(Runtime, integer_f32)	 // NOLINT
{
  auto integerFn = [](float value) -> float {
    return NAME_MANGLED(integer, float, float)(value);
  };

  EXPECT_NEAR(integerFn(-3.14), -4, 0.000001);
  EXPECT_NEAR(integerFn(3.14), 3, 0.000001);
}

TEST(Runtime, integer_f64)	 // NOLINT
{
  auto integerFn = [](double value) -> double {
    return NAME_MANGLED(integer, double, double)(value);
  };

  EXPECT_NEAR(integerFn(-3.14), -4, 0.000001);
  EXPECT_NEAR(integerFn(3.14), 3, 0.000001);
}

TEST(Runtime, linspace_i1)	 // NOLINT
{
	std::array<bool, 4> data = { true, false, false, false };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	double start = 0;
	double end = 1;

	NAME_MANGLED(linspace, void, ARRAY(bool), double, double)(&unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(data[i], (start + i * (end - start) / (data.size() - 1)) > 0);
  }
}

TEST(Runtime, linspace_i32)	 // NOLINT
{
	std::array<int32_t, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<int32_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(int32_t), double, double)(&unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(data[i], (int32_t) (start + i * (end - start) / (data.size() - 1)));
  }
}

TEST(Runtime, linspace_i64)	 // NOLINT
{
	std::array<int64_t, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<int64_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(int64_t), double, double)(&unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(data[i], (int64_t) (start + i * (end - start) / (data.size() - 1)));
  }
}

TEST(Runtime, linspace_f32)	 // NOLINT
{
	std::array<float, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(float), double, double)(&unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(data[i], start + i * (end - start) / (data.size() - 1));
  }
}

TEST(Runtime, linspace_f64)	 // NOLINT
{
	std::array<double, 4> data = { -1, -1, -1, -1 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	double start = 0;
	double end = 2;

	NAME_MANGLED(linspace, void, ARRAY(double), double, double)(&unsized, start, end);

	for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(data[i], start + i * (end - start) / (data.size() - 1));
  }
}

TEST(Runtime, log_f32)	 // NOLINT
{
  auto logFn = [](float value) -> float {
    return NAME_MANGLED(log, float, float)(value);
  };

	EXPECT_NEAR(logFn(1), 0, 0.000001);
	EXPECT_NEAR(logFn(2.718281828), 1, 0.000001);
	EXPECT_NEAR(logFn(7.389056099), 2, 0.000001);
	EXPECT_NEAR(logFn(0.367879441), -1, 0.000001);
}

TEST(Runtime, log_f64)	 // NOLINT
{
  auto logFn = [](double value) -> double {
    return NAME_MANGLED(log, double, double)(value);
  };

	EXPECT_NEAR(logFn(1), 0, 0.000001);
	EXPECT_NEAR(logFn(2.718281828), 1, 0.000001);
	EXPECT_NEAR(logFn(7.389056099), 2, 0.000001);
	EXPECT_NEAR(logFn(0.367879441), -1, 0.000001);
}

TEST(Runtime, log10_f32)	 // NOLINT
{
  auto log10Fn = [](float value) -> float {
    return NAME_MANGLED(log10, float, float)(value);
  };

	EXPECT_NEAR(log10Fn(1), 0, 0.000001);
	EXPECT_NEAR(log10Fn(10), 1, 0.000001);
	EXPECT_NEAR(log10Fn(100), 2, 0.000001);
	EXPECT_NEAR(log10Fn(0.1), -1, 0.000001);
}

TEST(Runtime, log10_f64)	 // NOLINT
{
  auto log10Fn = [](double value) -> double {
    return NAME_MANGLED(log10, double, double)(value);
  };

	EXPECT_NEAR(log10Fn(1), 0, 0.000001);
	EXPECT_NEAR(log10Fn(10), 1, 0.000001);
	EXPECT_NEAR(log10Fn(100), 2, 0.000001);
	EXPECT_NEAR(log10Fn(0.1), -1, 0.000001);
}

TEST(Runtime, max_ai1)	 // NOLINT
{
	std::array<bool, 4> data = { false, true, true, false };
	ArraySizes<bool, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	auto result = NAME_MANGLED(max, bool, ARRAY(bool))(&unsized);
	EXPECT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_ai32)	 // NOLINT
{
	std::array<int32_t, 4> data = { 5, 0, -3, 2 };
	ArraySizes<int32_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	auto result = NAME_MANGLED(max, int32_t, ARRAY(int32_t))(&unsized);
	EXPECT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_ai64)	 // NOLINT
{
	std::array<int64_t, 4> data = { 5, 0, -3, 2 };
	ArraySizes<int64_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	auto result = NAME_MANGLED(max, int64_t, ARRAY(int64_t))(&unsized);
	EXPECT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_af32)	 // NOLINT
{
	std::array<float, 4> data = { 5, 0, -3, 2 };
	ArraySizes<float, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	auto result = NAME_MANGLED(max, float, ARRAY(float))(&unsized);
	EXPECT_FLOAT_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_af64)	 // NOLINT
{
	std::array<double, 4> data = { 5, 0, -3, 2 };
	ArraySizes<double, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	auto result = NAME_MANGLED(max, double, ARRAY(double))(&unsized);
	EXPECT_DOUBLE_EQ(result, *std::max_element(data.begin(), data.end()));
}

TEST(Runtime, max_i1_i1)	 // NOLINT
{
	std::array<bool, 4> x = { false, false, true, true };
	std::array<bool, 4> y = { false, true, false, true };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(max, bool, bool, bool)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, max_i32_i32)	 // NOLINT
{
	std::array<int32_t, 3> x = { 0, 1, 2 };
	std::array<int32_t, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(max, int32_t, int32_t, int32_t)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, max_i64_i64)	 // NOLINT
{
	std::array<int64_t, 3> x = { 0, 1, 2 };
	std::array<int64_t, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(max, int64_t, int64_t, int64_t)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, max_f32_f32)	 // NOLINT
{
	std::array<float, 3> x = { 0, 1, 2 };
	std::array<float, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(max, float, float, float)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, max_f64_f64)	 // NOLINT
{
	std::array<double, 3> x = { 0, 1, 2 };
	std::array<double, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(max, double, double, double)(x, y);
		ASSERT_EQ(result, std::max(x, y));
	}
}

TEST(Runtime, min_ai1)	 // NOLINT
{
	std::array<bool, 4> data = { false, true, true, false };
	ArraySizes<bool, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	auto result = NAME_MANGLED(min, bool, ARRAY(bool))(&unsized);
	EXPECT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_ai32)	 // NOLINT
{
	std::array<int32_t, 4> data = { 5, 0, -3, 2 };
	ArraySizes<int32_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	auto result = NAME_MANGLED(min, int32_t, ARRAY(int32_t))(&unsized);
	EXPECT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_ai64)	 // NOLINT
{
	std::array<int64_t, 4> data = { 5, 0, -3, 2 };
	ArraySizes<int64_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	auto result = NAME_MANGLED(min, int64_t, ARRAY(int64_t))(&unsized);
	EXPECT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_af32)	 // NOLINT
{
	std::array<float, 4> data = { 5, 0, -3, 2 };
	ArraySizes<float, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	auto result = NAME_MANGLED(min, float, ARRAY(float))(&unsized);
	EXPECT_FLOAT_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_af64)	 // NOLINT
{
	std::array<double, 4> data = { 5, 0, -3, 2 };
	ArraySizes<double, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	auto result = NAME_MANGLED(min, double, ARRAY(double))(&unsized);
	EXPECT_DOUBLE_EQ(result, *std::min_element(data.begin(), data.end()));
}

TEST(Runtime, min_i1_i1)	 // NOLINT
{
	std::array<bool, 4> x = { false, false, true, true };
	std::array<bool, 4> y = { false, true, false, true };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(min, bool, bool, bool)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, min_i32_i32)	 // NOLINT
{
	std::array<int32_t, 3> x = { 0, 1, 2 };
	std::array<int32_t, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(min, int32_t, int32_t, int32_t)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, min_i64_i64)	 // NOLINT
{
	std::array<int64_t, 3> x = { 0, 1, 2 };
	std::array<int64_t, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(min, int64_t, int64_t, int64_t)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, min_f32_f32)	 // NOLINT
{
	std::array<float, 3> x = { 0, 1, 2 };
	std::array<float, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(min, float, float, float)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, min_f64_f64)	 // NOLINT
{
	std::array<double, 3> x = { 0, 1, 2 };
	std::array<double, 4> y = { 0, 2, 1 };

	for (const auto& [x, y] : llvm::zip(x, y)) {
		auto result = NAME_MANGLED(min, double, double, double)(x, y);
		ASSERT_EQ(result, std::min(x, y));
	}
}

TEST(Runtime, mod_i1_i1)	 // NOLINT
{
  auto modFn = [](bool x, bool y) -> bool {
    return NAME_MANGLED(mod, bool, bool, bool)(x, y);
  };

  EXPECT_EQ(modFn(false, true), false);
  EXPECT_EQ(modFn(true, true), false);
}

TEST(Runtime, mod_i32_i32)	 // NOLINT
{
  auto modFn = [](int32_t x, int32_t y) -> int32_t {
    return NAME_MANGLED(mod, int32_t, int32_t, int32_t)(x, y);
  };

  EXPECT_EQ(modFn(6, 3), 0);
  EXPECT_EQ(modFn(8, 3), 2);
  EXPECT_EQ(modFn(10, -3), -2);
  EXPECT_EQ(modFn(-10, 3), 2);
}

TEST(Runtime, mod_i64_i64)	 // NOLINT
{
  auto modFn = [](int64_t x, int64_t y) -> int64_t {
    return NAME_MANGLED(mod, int64_t, int64_t, int64_t)(x, y);
  };

  EXPECT_EQ(modFn(6, 3), 0);
  EXPECT_EQ(modFn(8, 3), 2);
  EXPECT_EQ(modFn(10, -3), -2);
  EXPECT_EQ(modFn(-10, 3), 2);
}

TEST(Runtime, mod_f32_f32)	 // NOLINT
{
  auto modFn = [](float x, float y) -> float {
    return NAME_MANGLED(mod, float, float, float)(x, y);
  };

  EXPECT_NEAR(modFn(6, 3), 0, 0.000001);
  EXPECT_NEAR(modFn(8.5, 3), 2.5, 0.000001);
  EXPECT_NEAR(modFn(3, 1.4), 0.2, 0.000001);
  EXPECT_NEAR(modFn(-3, 1.4), 1.2, 0.000001);
  EXPECT_NEAR(modFn(3, -1.4), -1.2, 0.000001);
}

TEST(Runtime, mod_f64_f64)	 // NOLINT
{
  auto modFn = [](double x, double y) -> double {
    return NAME_MANGLED(mod, double, double, double)(x, y);
  };

  EXPECT_NEAR(modFn(6, 3), 0, 0.000001);
  EXPECT_NEAR(modFn(8.5, 3), 2.5, 0.000001);
  EXPECT_NEAR(modFn(3, 1.4), 0.2, 0.000001);
  EXPECT_NEAR(modFn(-3, 1.4), 1.2, 0.000001);
  EXPECT_NEAR(modFn(3, -1.4), -1.2, 0.000001);
}

TEST(Runtime, ones_i1)	 // NOLINT
{
	std::array<bool, 4> data = { false, false, false, false };
	ArraySizes<bool, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(bool))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == true;
  }));
}

TEST(Runtime, ones_i32)	 // NOLINT
{
	std::array<int32_t, 4> data = { 0, 0, 0, 0 };
	ArraySizes<int32_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(int32_t))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == 1;
  }));
}

TEST(Runtime, ones_i64)	 // NOLINT
{
	std::array<int64_t, 4> data = { 0, 0, 0, 0 };
	ArraySizes<int64_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(int64_t))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == 1;
  }));
}

TEST(Runtime, ones_f32)	 // NOLINT
{
	std::array<float, 4> data = { 0, 0, 0, 0 };
	ArraySizes<float, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(float))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == 1;
  }));
}

TEST(Runtime, ones_f64)	 // NOLINT
{
	std::array<double, 4> data = { 0, 0, 0, 0 };
	ArraySizes<double, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	NAME_MANGLED(ones, void, ARRAY(double))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == 1;
  }));
}

TEST(Runtime, product_ai1)	 // NOLINT
{
	std::array<bool, 3> data = { false, true, true };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);
	auto result = NAME_MANGLED(product, bool, ARRAY(bool))(&unsized);
	EXPECT_EQ(result, (bool) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_ai32)	 // NOLINT
{
	std::array<int32_t, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);
	auto result = NAME_MANGLED(product, int32_t, ARRAY(int32_t))(&unsized);
	EXPECT_EQ(result, (int) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_ai64)	 // NOLINT
{
	std::array<int64_t, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);
	auto result = NAME_MANGLED(product, int64_t, ARRAY(int64_t))(&unsized);
	EXPECT_EQ(result, (long) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_af32)	 // NOLINT
{
	std::array<float, 3> data = { 1, 2, 3 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);
	auto result = NAME_MANGLED(product, float, ARRAY(float))(&unsized);
	EXPECT_FLOAT_EQ(result, (float) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, product_af64)	 // NOLINT
{
	std::array<double, 3> data = { 1, 2, 3 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);
	auto result = NAME_MANGLED(product, double, ARRAY(double))(&unsized);
	EXPECT_DOUBLE_EQ(result, (double) std::accumulate(data.begin(), data.end(), 1, std::multiplies<>()));
}

TEST(Runtime, rem_i1_i1)	 // NOLINT
{
  auto modFn = [](bool x, bool y) -> bool {
    return NAME_MANGLED(rem, bool, bool, bool)(x, y);
  };

  EXPECT_EQ(modFn(false, true), false);
  EXPECT_EQ(modFn(true, true), false);
}

TEST(Runtime, rem_i32_i32)	 // NOLINT
{
  auto modFn = [](int32_t x, int32_t y) -> int32_t {
    return NAME_MANGLED(rem, int32_t, int32_t, int32_t)(x, y);
  };

  EXPECT_EQ(modFn(6, 3), 0);
  EXPECT_EQ(modFn(8, 3), 2);
  EXPECT_EQ(modFn(10, -3), 1);
  EXPECT_EQ(modFn(-10, 3), -1);
}

TEST(Runtime, rem_i64_i64)	 // NOLINT
{
  auto modFn = [](int64_t x, int64_t y) -> int64_t {
    return NAME_MANGLED(rem, int64_t, int64_t, int64_t)(x, y);
  };

  EXPECT_EQ(modFn(6, 3), 0);
  EXPECT_EQ(modFn(8, 3), 2);
  EXPECT_EQ(modFn(10, -3), 1);
  EXPECT_EQ(modFn(-10, 3), -1);
}

TEST(Runtime, rem_f32_f32)	 // NOLINT
{
  auto modFn = [](float x, float y) -> float {
    return NAME_MANGLED(rem, float, float, float)(x, y);
  };

  EXPECT_NEAR(modFn(6, 3), 0, 0.000001);
  EXPECT_NEAR(modFn(8.5, 3), 2.5, 0.000001);
  EXPECT_NEAR(modFn(3, 1.4), 0.2, 0.000001);
  EXPECT_NEAR(modFn(-3, 1.4), -0.2, 0.000001);
  EXPECT_NEAR(modFn(3, -1.4), 0.2, 0.000001);
}

TEST(Runtime, rem_f64_f64)	 // NOLINT
{
  auto modFn = [](double x, double y) -> double {
    return NAME_MANGLED(rem, double, double, double)(x, y);
  };

  EXPECT_NEAR(modFn(6, 3), 0, 0.000001);
  EXPECT_NEAR(modFn(8.5, 3), 2.5, 0.000001);
  EXPECT_NEAR(modFn(3, 1.4), 0.2, 0.000001);
  EXPECT_NEAR(modFn(-3, 1.4), -0.2, 0.000001);
  EXPECT_NEAR(modFn(3, -1.4), 0.2, 0.000001);
}

TEST(Runtime, sign_i1)	 // NOLINT
{
  auto signI32Fn = [](bool value) -> int32_t {
    return NAME_MANGLED(sign, int32_t, bool)(value);
  };

  auto signI64Fn = [](bool value) -> int64_t {
    return NAME_MANGLED(sign, int64_t, bool)(value);
  };

	EXPECT_EQ(signI32Fn(false), 0);
	EXPECT_EQ(signI64Fn(false), 0);

	EXPECT_EQ(signI32Fn(true), 1);
	EXPECT_EQ(signI64Fn(true), 1);
}

TEST(Runtime, sign_i32)	 // NOLINT
{
  auto signI32Fn = [](int32_t value) -> int32_t {
    return NAME_MANGLED(sign, int32_t, int32_t)(value);
  };

  auto signI64Fn = [](int32_t value) -> int64_t {
    return NAME_MANGLED(sign, int64_t, int32_t)(value);
  };

	EXPECT_EQ(signI32Fn(-2), -1);
	EXPECT_EQ(signI64Fn(-2), -1);

	EXPECT_EQ(signI32Fn(0), 0);
	EXPECT_EQ(signI64Fn(0), 0);

	EXPECT_EQ(signI32Fn(2), 1);
	EXPECT_EQ(signI64Fn(2), 1);
}

TEST(Runtime, sign_i64)	 // NOLINT
{
  auto signI32Fn = [](int64_t value) -> int32_t {
    return NAME_MANGLED(sign, int32_t, int64_t)(value);
  };

  auto signI64Fn = [](int64_t value) -> int64_t {
    return NAME_MANGLED(sign, int64_t, int64_t)(value);
  };

	EXPECT_EQ(signI32Fn(-2), -1);
	EXPECT_EQ(signI64Fn(-2), -1);

	EXPECT_EQ(signI32Fn(0), 0);
	EXPECT_EQ(signI64Fn(0), 0);

	EXPECT_EQ(signI32Fn(2), 1);
	EXPECT_EQ(signI64Fn(2), 1);
}

TEST(Runtime, sign_f32)	 // NOLINT
{
  auto signI32Fn = [](float value) -> int32_t {
    return NAME_MANGLED(sign, int32_t, float)(value);
  };

  auto signI64Fn = [](float value) -> int64_t {
    return NAME_MANGLED(sign, int64_t, float)(value);
  };

	EXPECT_EQ(signI32Fn(-2), -1);
	EXPECT_EQ(signI64Fn(-2), -1);

	EXPECT_EQ(signI32Fn(0), 0);
	EXPECT_EQ(signI64Fn(0), 0);

	EXPECT_EQ(signI32Fn(2), 1);
	EXPECT_EQ(signI64Fn(2), 1);
}

TEST(Runtime, sign_f64)	 // NOLINT
{
  auto signI32Fn = [](double value) -> int32_t {
    return NAME_MANGLED(sign, int32_t, double)(value);
  };

  auto signI64Fn = [](double value) -> int64_t {
    return NAME_MANGLED(sign, int64_t, double)(value);
  };

	EXPECT_EQ(signI32Fn(-2), -1);
	EXPECT_EQ(signI64Fn(-2), -1);

	EXPECT_EQ(signI32Fn(0), 0);
	EXPECT_EQ(signI64Fn(0), 0);

	EXPECT_EQ(signI32Fn(2), 1);
	EXPECT_EQ(signI64Fn(2), 1);
}

TEST(Runtime, sin_f32)	 // NOLINT
{
  auto sinFn = [](float value) -> float {
    return NAME_MANGLED(sin, float, float)(value);
  };

	EXPECT_NEAR(sinFn(0), 0, 0.000001);
	EXPECT_NEAR(sinFn(M_PI / 6), 0.5, 0.000001);
	EXPECT_NEAR(sinFn(M_PI / 4), 0.707106781, 0.000001);
	EXPECT_NEAR(sinFn(M_PI / 2), 1, 0.000001);
	EXPECT_NEAR(sinFn(M_PI), 0, 0.000001);
	EXPECT_NEAR(sinFn(2 * M_PI), 0, 0.000001);
}

TEST(Runtime, sin_f64)	 // NOLINT
{
  auto sinFn = [](double value) -> double {
    return NAME_MANGLED(sin, double, double)(value);
  };

	EXPECT_NEAR(sinFn(0), 0, 0.000001);
	EXPECT_NEAR(sinFn(M_PI / 6), 0.5, 0.000001);
	EXPECT_NEAR(sinFn(M_PI / 4), 0.707106781, 0.000001);
	EXPECT_NEAR(sinFn(M_PI / 2), 1, 0.000001);
	EXPECT_NEAR(sinFn(M_PI), 0, 0.000001);
	EXPECT_NEAR(sinFn(2 * M_PI), 0, 0.000001);
}

TEST(Runtime, sqrt_f32)	 // NOLINT
{
  auto sqrtFn = [](float value) -> float {
    return NAME_MANGLED(sqrt, float, float)(value);
  };

	EXPECT_FLOAT_EQ(sqrtFn(0), 0);
	EXPECT_FLOAT_EQ(sqrtFn(1), 1);
	EXPECT_FLOAT_EQ(sqrtFn(4), 2);
}

TEST(Runtime, sqrt_f64)	 // NOLINT
{
  auto sqrtFn = [](double value) -> double {
    return NAME_MANGLED(sqrt, double, double)(value);
  };

	EXPECT_DOUBLE_EQ(sqrtFn(0), 0);
	EXPECT_DOUBLE_EQ(sqrtFn(1), 1);
	EXPECT_DOUBLE_EQ(sqrtFn(4), 2);
}

TEST(Runtime, sinh_f32)	 // NOLINT
{
  auto sinhFn = [](float value) -> float {
    return NAME_MANGLED(sinh, float, float)(value);
  };

	EXPECT_NEAR(sinhFn(0), 0, 0.000001);
	EXPECT_NEAR(sinhFn(1), 1.175201193, 0.000001);
}

TEST(Runtime, sinh_f64)	 // NOLINT
{
  auto sinhFn = [](double value) -> double {
    return NAME_MANGLED(sinh, double, double)(value);
  };

	EXPECT_NEAR(sinhFn(0), 0, 0.000001);
	EXPECT_NEAR(sinhFn(1), 1.175201193, 0.000001);
}

TEST(Runtime, sum_ai1)	 // NOLINT
{
	std::array<bool, 3> data = { false, true, true };
	ArrayDescriptor<bool, 1> descriptor(data);
	UnsizedArrayDescriptor<bool> unsized(descriptor);
	auto result = NAME_MANGLED(sum, bool, ARRAY(bool))(&unsized);
	EXPECT_EQ(result, (bool) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_ai32)	 // NOLINT
{
	std::array<int32_t, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int32_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);
	auto result = NAME_MANGLED(sum, int32_t, ARRAY(int32_t))(&unsized);
	EXPECT_EQ(result, (int32_t) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_ai64)	 // NOLINT
{
	std::array<int64_t, 3> data = { 1, 2, 3 };
	ArrayDescriptor<int64_t, 1> descriptor(data);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);
	auto result = NAME_MANGLED(sum, int64_t, ARRAY(int64_t))(&unsized);
	EXPECT_EQ(result, (int64_t) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_af32)	 // NOLINT
{
	std::array<float, 3> data = { 1, 2, 3 };
	ArrayDescriptor<float, 1> descriptor(data);
	UnsizedArrayDescriptor<float> unsized(descriptor);
	auto result = NAME_MANGLED(sum, float, ARRAY(float))(&unsized);
	EXPECT_FLOAT_EQ(result, (float) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_af64)	 // NOLINT
{
	std::array<double, 3> data = { 1, 2, 3 };
	ArrayDescriptor<double, 1> descriptor(data);
	UnsizedArrayDescriptor<double> unsized(descriptor);
	auto result = NAME_MANGLED(sum, double, ARRAY(double))(&unsized);
	EXPECT_DOUBLE_EQ(result, (double) std::accumulate(data.begin(), data.end(), 0, std::plus<>()));
}

TEST(Runtime, symmetric_ai1_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai1_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai1_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai1_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai1_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<bool, 9> destination = { true, false, true, true, false, true, true, false, true };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (bool) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai32_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai32_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai32_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai32_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai32_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int32_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai64_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int64_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai64_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int32_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai64_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int64_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai64_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int64_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_ai64_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<int64_t, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (int64_t) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af32_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af32_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af32_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af32_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af32_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<float, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (float) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af64_ai1)	 // NOLINT
{
	std::array<bool, 9> source = { true, false, true, true, false, true, true, false, true };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af64_ai32)	 // NOLINT
{
	std::array<int32_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af64_ai64)	 // NOLINT
{
	std::array<int64_t, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af64_af32)	 // NOLINT
{
	std::array<float, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, symmetric_af64_af64)	 // NOLINT
{
	std::array<double, 9> source = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
	std::array<double, 9> destination = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 3 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = i; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(i, j), (double) sourceDescriptor.get(i, j));
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, tan_f32)	 // NOLINT
{
	std::array<float, 5> data = { 0, M_PI / 6, M_PI / 4, M_PI, 2 * M_PI };

  auto tanFn = [](float value) -> float {
    return NAME_MANGLED(tan, float, float)(value);
  };

	EXPECT_NEAR(tanFn(data[0]), 0, 0.000001);
	EXPECT_NEAR(tanFn(data[1]), 0.577350269, 0.000001);
	EXPECT_NEAR(tanFn(data[2]), 1, 0.000001);
	EXPECT_NEAR(tanFn(data[3]), 0, 0.000001);
	EXPECT_NEAR(tanFn(data[4]), 0, 0.000001);
}

TEST(Runtime, tan_f64)	 // NOLINT
{
	std::array<double, 5> data = { 0, M_PI / 6, M_PI / 4, M_PI, 2 * M_PI };

  auto tanFn = [](double value) -> double {
    return NAME_MANGLED(tan, double, double)(value);
  };

	EXPECT_NEAR(tanFn(data[0]), 0, 0.000000001);
	EXPECT_NEAR(tanFn(data[1]), 0.577350269, 0.000000001);
	EXPECT_NEAR(tanFn(data[2]), 1, 0.000000001);
	EXPECT_NEAR(tanFn(data[3]), 0, 0.000000001);
	EXPECT_NEAR(tanFn(data[4]), 0, 0.000000001);
}

TEST(Runtime, tanh_f32)	 // NOLINT
{
  auto tanhFn = [](float value) -> float {
    return NAME_MANGLED(tanh, float, float)(value);
  };

	EXPECT_NEAR(tanhFn(0), 0, 0.000001);
	EXPECT_NEAR(tanhFn(1), 0.761594155, 0.000001);
}

TEST(Runtime, tanh_f64)	 // NOLINT
{
  auto tanhFn = [](double value) -> double {
    return NAME_MANGLED(tanh, double, double)(value);
  };

	EXPECT_NEAR(tanhFn(0), 0, 0.000001);
	EXPECT_NEAR(tanhFn(1), 0.761594155, 0.000001);
}

TEST(Runtime, transpose_ai1_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai1_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai1_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai1_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai1_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<bool, 6> destination = { true, false, true, false, true, false };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<bool, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<bool> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (bool) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai32_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i)
		for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j)
			EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
}

TEST(Runtime, transpose_ai32_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai32_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai32_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int32_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai32_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int32_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int32_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (int32_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai64_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai64_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai64_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai64_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_ai64_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<int64_t, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<int64_t, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<int64_t> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (int64_t) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af32_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af32_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af32_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af32_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af32_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<float, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<float, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<float> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (float) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af64_ai1)	 // NOLINT
{
	std::array<bool, 6> source = { false, false, false, true, true, true };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<bool, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<bool> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(bool))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af64_ai32)	 // NOLINT
{
	std::array<int32_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int32_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int32_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(int32_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af64_ai64)	 // NOLINT
{
	std::array<int64_t, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<int64_t, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<int64_t> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(int64_t))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af64_af32)	 // NOLINT
{
	std::array<float, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<float, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<float> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(float))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, transpose_af64_af64)	 // NOLINT
{
	std::array<double, 6> source = { 0, 0, 0, 1, 1, 1 };
	std::array<double, 6> destination = { 1, 0, 1, 0, 1, 0 };

	ArrayDescriptor<double, 2> sourceDescriptor(source.data(), { 2, 3 });
	UnsizedArrayDescriptor<double> unsizedSource(sourceDescriptor);

	ArrayDescriptor<double, 2> destinationDescriptor(destination.data(), { 3, 2 });
	UnsizedArrayDescriptor<double> unsizedDestination(destinationDescriptor);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(double))(&unsizedDestination, &unsizedSource);

	for (size_t i = 0; i < sourceDescriptor.getDimension(0); ++i) {
    for (size_t j = 0; j < sourceDescriptor.getDimension(1); ++j) {
      EXPECT_EQ(destinationDescriptor.get(j, i), (double) sourceDescriptor.get(i, j));
    }
  }
}

TEST(Runtime, zeros_i1)	 // NOLINT
{
	std::array<bool, 4> data = { true, true, true, true };
	ArraySizes<bool, 2> sizes = { 2, 2 };

	ArrayDescriptor<bool, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<bool> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(bool))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == false;
  }));
}

TEST(Runtime, zeros_i32)	 // NOLINT
{
	std::array<int32_t, 4> data = { 1, 1, 1, 1 };
	ArraySizes<int32_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int32_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int32_t> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(int32_t))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == 0;
  }));
}

TEST(Runtime, zeros_i64)	 // NOLINT
{
	std::array<int64_t, 4> data = { 1, 1, 1, 1 };
	ArraySizes<int64_t, 2> sizes = { 2, 2 };

	ArrayDescriptor<int64_t, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<int64_t> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(int64_t))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == 0;
  }));
}

TEST(Runtime, zeros_f32)	 // NOLINT
{
	std::array<float, 4> data = { 1, 1, 1, 1 };
	ArraySizes<float, 2> sizes = { 2, 2 };

	ArrayDescriptor<float, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<float> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(float))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == 0;
  }));
}

TEST(Runtime, zeros_f64)	 // NOLINT
{
	std::array<double, 4> data = { 1, 1, 1, 1 };
	ArraySizes<double, 2> sizes = { 2, 2 };

	ArrayDescriptor<double, 2> descriptor(data.data(), sizes);
	UnsizedArrayDescriptor<double> unsized(descriptor);

	NAME_MANGLED(zeros, void, ARRAY(double))(&unsized);

  EXPECT_TRUE(llvm::all_of(data, [](const auto& element) {
    return element == 0;
  }));
}
