#include "marco/Runtime/BuiltInFunctions.h"
#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include <cmath>
#include <numeric>

#include "Utils.h"

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

TEST(Runtime, ceil_i1)
{
  auto ceilFn = [](bool value) -> bool {
    return NAME_MANGLED(ceil, bool, bool)(value);
  };

  EXPECT_EQ(ceilFn(false), false);
  EXPECT_EQ(ceilFn(true), true);
}

TEST(Runtime, ceil_i32)
{
  auto ceilFn = [](int32_t value) -> int32_t {
    return NAME_MANGLED(ceil, int32_t, int32_t)(value);
  };

  EXPECT_EQ(ceilFn(-3), -3);
  EXPECT_EQ(ceilFn(3), 3);
}

TEST(Runtime, ceil_i64)
{
  auto ceilFn = [](int64_t value) -> int64_t {
    return NAME_MANGLED(ceil, int64_t, int64_t)(value);
  };

  EXPECT_EQ(ceilFn(-3), -3);
  EXPECT_EQ(ceilFn(3), 3);
}

TEST(Runtime, ceil_f32)
{
  auto ceilFn = [](float value) -> float {
    return NAME_MANGLED(ceil, float, float)(value);
  };

  EXPECT_NEAR(ceilFn(-3.14), -3, 0.000001);
  EXPECT_NEAR(ceilFn(3.14), 4, 0.000001);
}

TEST(Runtime, ceil_f64)
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
  std::array<bool, 9> destinationValues = { false, true, true, true, false, true, true, true, false };
  std::array<bool, 3> sourceValues = { true, true, true };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], false);
  EXPECT_EQ(destination[0][2], false);

  EXPECT_EQ(destination[1][0], false);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], false);

  EXPECT_EQ(destination[2][0], false);
  EXPECT_EQ(destination[2][1], false);
  EXPECT_EQ(destination[2][2], true);
}

TEST(Runtime, diagonalSquareMatrix_i1_i32)
{
  std::array<bool, 9> destinationValues = { false, true, true, true, false, true, true, true, false };
  std::array<int32_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], false);
  EXPECT_EQ(destination[0][2], false);

  EXPECT_EQ(destination[1][0], false);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], false);

  EXPECT_EQ(destination[2][0], false);
  EXPECT_EQ(destination[2][1], false);
  EXPECT_EQ(destination[2][2], true);
}

TEST(Runtime, diagonalSquareMatrix_i1_i64)
{
  std::array<bool, 9> destinationValues = { false, true, true, true, false, true, true, true, false };
  std::array<int64_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], false);
  EXPECT_EQ(destination[0][2], false);

  EXPECT_EQ(destination[1][0], false);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], false);

  EXPECT_EQ(destination[2][0], false);
  EXPECT_EQ(destination[2][1], false);
  EXPECT_EQ(destination[2][2], true);
}

TEST(Runtime, diagonalSquareMatrix_i1_f32)
{
  std::array<bool, 9> destinationValues = { false, true, true, true, false, true, true, true, false };
  std::array<float, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], false);
  EXPECT_EQ(destination[0][2], false);

  EXPECT_EQ(destination[1][0], false);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], false);

  EXPECT_EQ(destination[2][0], false);
  EXPECT_EQ(destination[2][1], false);
  EXPECT_EQ(destination[2][2], true);
}

TEST(Runtime, diagonalSquareMatrix_i1_f64)
{
  std::array<bool, 9> destinationValues = { false, true, true, true, false, true, true, true, false };
  std::array<double, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(bool), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], false);
  EXPECT_EQ(destination[0][2], false);

  EXPECT_EQ(destination[1][0], false);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], false);

  EXPECT_EQ(destination[2][0], false);
  EXPECT_EQ(destination[2][1], false);
  EXPECT_EQ(destination[2][2], true);
}

TEST(Runtime, diagonalSquareMatrix_i32_i1)
{
  std::array<int32_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<bool, 3> sourceValues = { true, true, true };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 1);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 1);
}

TEST(Runtime, diagonalSquareMatrix_i32_i32)
{
  std::array<int32_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<int32_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 2);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_i32_i64)
{
  std::array<int32_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<int64_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 2);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_i32_f32)
{
  std::array<int32_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<float, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 2);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_i32_f64)
{
  std::array<int32_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<double, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int32_t), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 2);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_i64_i1)
{
  std::array<int64_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<bool, 3> sourceValues = { true, true, true };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 1);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 1);
}

TEST(Runtime, diagonalSquareMatrix_i64_i32)
{
  std::array<int64_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<int32_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 2);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_i64_i64)
{
  std::array<int64_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<int64_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 2);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_i64_f32)
{
  std::array<int64_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<float, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 2);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_i64_f64)
{
  std::array<int64_t, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<double, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(int64_t), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 0);
  EXPECT_EQ(destination[0][2], 0);

  EXPECT_EQ(destination[1][0], 0);
  EXPECT_EQ(destination[1][1], 2);
  EXPECT_EQ(destination[1][2], 0);

  EXPECT_EQ(destination[2][0], 0);
  EXPECT_EQ(destination[2][1], 0);
  EXPECT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_f32_i1)
{
  std::array<float, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<bool, 3> sourceValues = { true, true, true };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 0);
  EXPECT_FLOAT_EQ(destination[0][2], 0);

  EXPECT_FLOAT_EQ(destination[1][0], 0);
  EXPECT_FLOAT_EQ(destination[1][1], 1);
  EXPECT_FLOAT_EQ(destination[1][2], 0);

  EXPECT_FLOAT_EQ(destination[2][0], 0);
  EXPECT_FLOAT_EQ(destination[2][1], 0);
  EXPECT_FLOAT_EQ(destination[2][2], 1);
}

TEST(Runtime, diagonalSquareMatrix_f32_i32)
{
  std::array<float, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<int32_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 0);
  EXPECT_FLOAT_EQ(destination[0][2], 0);

  EXPECT_FLOAT_EQ(destination[1][0], 0);
  EXPECT_FLOAT_EQ(destination[1][1], 2);
  EXPECT_FLOAT_EQ(destination[1][2], 0);

  EXPECT_FLOAT_EQ(destination[2][0], 0);
  EXPECT_FLOAT_EQ(destination[2][1], 0);
  EXPECT_FLOAT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_f32_i64)
{
  std::array<float, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<int64_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 0);
  EXPECT_FLOAT_EQ(destination[0][2], 0);

  EXPECT_FLOAT_EQ(destination[1][0], 0);
  EXPECT_FLOAT_EQ(destination[1][1], 2);
  EXPECT_FLOAT_EQ(destination[1][2], 0);

  EXPECT_FLOAT_EQ(destination[2][0], 0);
  EXPECT_FLOAT_EQ(destination[2][1], 0);
  EXPECT_FLOAT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_f32_f32)
{
  std::array<float, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<float, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 0);
  EXPECT_FLOAT_EQ(destination[0][2], 0);

  EXPECT_FLOAT_EQ(destination[1][0], 0);
  EXPECT_FLOAT_EQ(destination[1][1], 2);
  EXPECT_FLOAT_EQ(destination[1][2], 0);

  EXPECT_FLOAT_EQ(destination[2][0], 0);
  EXPECT_FLOAT_EQ(destination[2][1], 0);
  EXPECT_FLOAT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_f32_f64)
{
  std::array<float, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<double, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(float), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 0);
  EXPECT_FLOAT_EQ(destination[0][2], 0);

  EXPECT_FLOAT_EQ(destination[1][0], 0);
  EXPECT_FLOAT_EQ(destination[1][1], 2);
  EXPECT_FLOAT_EQ(destination[1][2], 0);

  EXPECT_FLOAT_EQ(destination[2][0], 0);
  EXPECT_FLOAT_EQ(destination[2][1], 0);
  EXPECT_FLOAT_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_f64_i1)
{
  std::array<double, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<bool, 3> sourceValues = { true, true, true };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 0);
  EXPECT_DOUBLE_EQ(destination[0][2], 0);

  EXPECT_DOUBLE_EQ(destination[1][0], 0);
  EXPECT_DOUBLE_EQ(destination[1][1], 1);
  EXPECT_DOUBLE_EQ(destination[1][2], 0);

  EXPECT_DOUBLE_EQ(destination[2][0], 0);
  EXPECT_DOUBLE_EQ(destination[2][1], 0);
  EXPECT_DOUBLE_EQ(destination[2][2], 1);
}

TEST(Runtime, diagonalSquareMatrix_f64_i32)
{
  std::array<double, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<int32_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 0);
  EXPECT_DOUBLE_EQ(destination[0][2], 0);

  EXPECT_DOUBLE_EQ(destination[1][0], 0);
  EXPECT_DOUBLE_EQ(destination[1][1], 2);
  EXPECT_DOUBLE_EQ(destination[1][2], 0);

  EXPECT_DOUBLE_EQ(destination[2][0], 0);
  EXPECT_DOUBLE_EQ(destination[2][1], 0);
  EXPECT_DOUBLE_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_f64_i64)
{
  std::array<double, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<int64_t, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 0);
  EXPECT_DOUBLE_EQ(destination[0][2], 0);

  EXPECT_DOUBLE_EQ(destination[1][0], 0);
  EXPECT_DOUBLE_EQ(destination[1][1], 2);
  EXPECT_DOUBLE_EQ(destination[1][2], 0);

  EXPECT_DOUBLE_EQ(destination[2][0], 0);
  EXPECT_DOUBLE_EQ(destination[2][1], 0);
  EXPECT_DOUBLE_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_f64_f32)
{
  std::array<double, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<float, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 0);
  EXPECT_DOUBLE_EQ(destination[0][2], 0);

  EXPECT_DOUBLE_EQ(destination[1][0], 0);
  EXPECT_DOUBLE_EQ(destination[1][1], 2);
  EXPECT_DOUBLE_EQ(destination[1][2], 0);

  EXPECT_DOUBLE_EQ(destination[2][0], 0);
  EXPECT_DOUBLE_EQ(destination[2][1], 0);
  EXPECT_DOUBLE_EQ(destination[2][2], 3);
}

TEST(Runtime, diagonalSquareMatrix_f64_f64)
{
  std::array<double, 9> destinationValues = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  std::array<double, 3> sourceValues = { 1, 2, 3 };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef(sourceValues);
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(diagonal, void, ARRAY(double), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 0);
  EXPECT_DOUBLE_EQ(destination[0][2], 0);

  EXPECT_DOUBLE_EQ(destination[1][0], 0);
  EXPECT_DOUBLE_EQ(destination[1][1], 2);
  EXPECT_DOUBLE_EQ(destination[1][2], 0);

  EXPECT_DOUBLE_EQ(destination[2][0], 0);
  EXPECT_DOUBLE_EQ(destination[2][1], 0);
  EXPECT_DOUBLE_EQ(destination[2][2], 3);
}

TEST(Runtime, div_i1_i1)
{
  auto divFn = [](bool x, bool y) -> bool {
    return NAME_MANGLED(div, bool, bool, bool)(x, y);
  };

  EXPECT_EQ(divFn(false, true), false);
  EXPECT_EQ(divFn(true, true), true);
}

TEST(Runtime, div_i32_i32)
{
  auto divFn = [](int32_t x, int32_t y) -> int32_t {
    return NAME_MANGLED(div, int32_t, int32_t, int32_t)(x, y);
  };

  EXPECT_EQ(divFn(6, 3), 2);
  EXPECT_EQ(divFn(8, 3), 2);
  EXPECT_EQ(divFn(10, -3), -3);
  EXPECT_EQ(divFn(-10, 3), -3);
}

TEST(Runtime, div_i64_i64)
{
  auto divFn = [](int64_t x, int64_t y) -> int64_t {
    return NAME_MANGLED(div, int64_t, int64_t, int64_t)(x, y);
  };

  EXPECT_EQ(divFn(6, 3), 2);
  EXPECT_EQ(divFn(8, 3), 2);
  EXPECT_EQ(divFn(10, -3), -3);
  EXPECT_EQ(divFn(-10, 3), -3);
}

TEST(Runtime, div_f32_f32)
{
  auto divFn = [](float x, float y) -> float {
    return NAME_MANGLED(div, float, float, float)(x, y);
  };

  EXPECT_NEAR(divFn(6, 3), 2, 0.000001);
  EXPECT_NEAR(divFn(8.5, 3), 2, 0.000001);
  EXPECT_NEAR(divFn(3, 1.4), 2, 0.000001);
  EXPECT_NEAR(divFn(-3, 1.4), -2, 0.000001);
  EXPECT_NEAR(divFn(3, -1.4), -2, 0.000001);
}

TEST(Runtime, div_f64_f64)
{
  auto divFn = [](double x, double y) -> double {
    return NAME_MANGLED(div, double, double, double)(x, y);
  };

  EXPECT_NEAR(divFn(6, 3), 2, 0.000001);
  EXPECT_NEAR(divFn(8.5, 3), 2, 0.000001);
  EXPECT_NEAR(divFn(3, 1.4), 2, 0.000001);
  EXPECT_NEAR(divFn(-3, 1.4), -2, 0.000001);
  EXPECT_NEAR(divFn(3, -1.4), -2, 0.000001);
}

TEST(Runtime, exp_f32)
{
  auto expFn = [](float exponent) -> float {
    return NAME_MANGLED(exp, float, float)(exponent);
  };

  EXPECT_NEAR(expFn(0), 1, 0.000001);
  EXPECT_NEAR(expFn(1), 2.718281, 0.000001);
  EXPECT_NEAR(expFn(2), 7.389056, 0.000001);
  EXPECT_NEAR(expFn(-2), 0.135335, 0.000001);
}

TEST(Runtime, exp_f64)
{
  auto expFn = [](double exponent) -> double {
    return NAME_MANGLED(exp, double, double)(exponent);
  };

  EXPECT_NEAR(expFn(0), 1, 0.000001);
  EXPECT_NEAR(expFn(1), 2.718281, 0.000001);
  EXPECT_NEAR(expFn(2), 7.389056, 0.000001);
  EXPECT_NEAR(expFn(-2), 0.135335, 0.000001);
}

TEST(Runtime, floor_i1)
{
  auto floorFn = [](bool value) -> bool {
    return NAME_MANGLED(floor, bool, bool)(value);
  };

  EXPECT_EQ(floorFn(false), false);
  EXPECT_EQ(floorFn(true), true);
}

TEST(Runtime, floor_i32)
{
  auto floorFn = [](int32_t value) -> int32_t {
    return NAME_MANGLED(floor, int32_t, int32_t)(value);
  };

  EXPECT_EQ(floorFn(-3), -3);
  EXPECT_EQ(floorFn(3), 3);
}

TEST(Runtime, floor_i64)
{
  auto floorFn = [](int64_t value) -> int64_t {
    return NAME_MANGLED(floor, int64_t, int64_t)(value);
  };

  EXPECT_EQ(floorFn(-3), -3);
  EXPECT_EQ(floorFn(3), 3);
}

TEST(Runtime, floor_f32)
{
  auto floorFn = [](float value) -> float {
    return NAME_MANGLED(floor, float, float)(value);
  };

  EXPECT_NEAR(floorFn(-3.14), -4, 0.000001);
  EXPECT_NEAR(floorFn(3.14), 3, 0.000001);
}

TEST(Runtime, floor_f64)
{
  auto floorFn = [](double value) -> double {
    return NAME_MANGLED(floor, double, double)(value);
  };

  EXPECT_NEAR(floorFn(-3.14), -4, 0.000001);
  EXPECT_NEAR(floorFn(3.14), 3, 0.000001);
}

TEST(Runtime, identitySquareMatrix_i1)
{
  std::array<bool, 9> values = { false, true, true, true, false, true, true, true, false };

  auto array = getMemRef<bool, 2>(values.data(), { 3, 3 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(identity, void, ARRAY(bool))(&unrankedArray);

  EXPECT_EQ(array[0][0], true);
  EXPECT_EQ(array[0][1], false);
  EXPECT_EQ(array[0][2], false);

  EXPECT_EQ(array[1][0], false);
  EXPECT_EQ(array[1][1], true);
  EXPECT_EQ(array[1][2], false);

  EXPECT_EQ(array[2][0], false);
  EXPECT_EQ(array[2][1], false);
  EXPECT_EQ(array[2][2], true);
}

TEST(Runtime, identitySquareMatrix_i32)
{
  std::array<int32_t, 9> values = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };

  auto array = getMemRef<int32_t, 2>(values.data(), { 3, 3 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(identity, void, ARRAY(int32_t))(&unrankedArray);

  EXPECT_EQ(array[0][0], 1);
  EXPECT_EQ(array[0][1], 0);
  EXPECT_EQ(array[0][2], 0);

  EXPECT_EQ(array[1][0], 0);
  EXPECT_EQ(array[1][1], 1);
  EXPECT_EQ(array[1][2], 0);

  EXPECT_EQ(array[2][0], 0);
  EXPECT_EQ(array[2][1], 0);
  EXPECT_EQ(array[2][2], 1);
}

TEST(Runtime, identitySquareMatrix_i64)
{
  std::array<int64_t, 9> values = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };

  auto array = getMemRef<int64_t, 2>(values.data(), { 3, 3 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(identity, void, ARRAY(int64_t))(&unrankedArray);

  EXPECT_EQ(array[0][0], 1);
  EXPECT_EQ(array[0][1], 0);
  EXPECT_EQ(array[0][2], 0);

  EXPECT_EQ(array[1][0], 0);
  EXPECT_EQ(array[1][1], 1);
  EXPECT_EQ(array[1][2], 0);

  EXPECT_EQ(array[2][0], 0);
  EXPECT_EQ(array[2][1], 0);
  EXPECT_EQ(array[2][2], 1);
}

TEST(Runtime, identitySquareMatrix_f32)
{
  std::array<float, 9> values = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };

  auto array = getMemRef<float, 2>(values.data(), { 3, 3 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(identity, void, ARRAY(float))(&unrankedArray);

  EXPECT_FLOAT_EQ(array[0][0], 1);
  EXPECT_FLOAT_EQ(array[0][1], 0);
  EXPECT_FLOAT_EQ(array[0][2], 0);

  EXPECT_FLOAT_EQ(array[1][0], 0);
  EXPECT_FLOAT_EQ(array[1][1], 1);
  EXPECT_FLOAT_EQ(array[1][2], 0);

  EXPECT_FLOAT_EQ(array[2][0], 0);
  EXPECT_FLOAT_EQ(array[2][1], 0);
  EXPECT_FLOAT_EQ(array[2][2], 1);
}

TEST(Runtime, identitySquareMatrix_f64)
{
  std::array<double, 9> values = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };

  auto array = getMemRef<double, 2>(values.data(), { 3, 3 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(identity, void, ARRAY(double))(&unrankedArray);

  EXPECT_DOUBLE_EQ(array[0][0], 1);
  EXPECT_DOUBLE_EQ(array[0][1], 0);
  EXPECT_DOUBLE_EQ(array[0][2], 0);

  EXPECT_DOUBLE_EQ(array[1][0], 0);
  EXPECT_DOUBLE_EQ(array[1][1], 1);
  EXPECT_DOUBLE_EQ(array[1][2], 0);

  EXPECT_DOUBLE_EQ(array[2][0], 0);
  EXPECT_DOUBLE_EQ(array[2][1], 0);
  EXPECT_DOUBLE_EQ(array[2][2], 1);
}

TEST(Runtime, integer_i1)
{
  auto integerFn = [](bool value) -> bool {
    return NAME_MANGLED(integer, bool, bool)(value);
  };

  EXPECT_EQ(integerFn(false), false);
  EXPECT_EQ(integerFn(true), true);
}

TEST(Runtime, integer_i32)
{
  auto integerFn = [](int32_t value) -> int32_t {
    return NAME_MANGLED(integer, int32_t, int32_t)(value);
  };

  EXPECT_EQ(integerFn(-3), -3);
  EXPECT_EQ(integerFn(3), 3);
}

TEST(Runtime, integer_i64)
{
  auto integerFn = [](int64_t value) -> int64_t {
    return NAME_MANGLED(integer, int64_t, int64_t)(value);
  };

  EXPECT_EQ(integerFn(-3), -3);
  EXPECT_EQ(integerFn(3), 3);
}

TEST(Runtime, integer_f32)
{
  auto integerFn = [](float value) -> float {
    return NAME_MANGLED(integer, float, float)(value);
  };

  EXPECT_NEAR(integerFn(-3.14), -4, 0.000001);
  EXPECT_NEAR(integerFn(3.14), 3, 0.000001);
}

TEST(Runtime, integer_f64)
{
  auto integerFn = [](double value) -> double {
    return NAME_MANGLED(integer, double, double)(value);
  };

  EXPECT_NEAR(integerFn(-3.14), -4, 0.000001);
  EXPECT_NEAR(integerFn(3.14), 3, 0.000001);
}

TEST(Runtime, linspace_i1)
{
  std::array<bool, 4> data = { true, false, false, false };

  auto array = getMemRef(data);
  auto unrankedArray = getUnrankedMemRef(array);

  double start = 0;
  double end = 1;

  NAME_MANGLED(linspace, void, ARRAY(bool), double, double)(&unrankedArray, start, end);

  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(array[i], (start + i * (end - start) / (data.size() - 1)) > 0);
  }
}

TEST(Runtime, linspace_i32)
{
  std::array<int32_t, 4> data = { -1, -1, -1, -1 };

  auto array = getMemRef(data);
  auto unrankedArray = getUnrankedMemRef(array);

  double start = 0;
  double end = 2;

  NAME_MANGLED(linspace, void, ARRAY(int32_t), double, double)(&unrankedArray, start, end);

  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(array[i], static_cast<int32_t>((start + i * (end - start) / (data.size() - 1))));
  }
}

TEST(Runtime, linspace_i64)
{
  std::array<int64_t, 4> data = { -1, -1, -1, -1 };

  auto array = getMemRef(data);
  auto unrankedArray = getUnrankedMemRef(array);

  double start = 0;
  double end = 2;

  NAME_MANGLED(linspace, void, ARRAY(int64_t), double, double)(&unrankedArray, start, end);

  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(array[i], static_cast<int64_t>((start + i * (end - start) / (data.size() - 1))));
  }
}

TEST(Runtime, linspace_f32)
{
  std::array<float, 4> data = { -1, -1, -1, -1 };

  auto array = getMemRef(data);
  auto unrankedArray = getUnrankedMemRef(array);

  double start = 0;
  double end = 2;

  NAME_MANGLED(linspace, void, ARRAY(float), double, double)(&unrankedArray, start, end);

  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(array[i], start + i * (end - start) / (data.size() - 1));
  }
}

TEST(Runtime, linspace_f64)
{
  std::array<double, 4> data = { -1, -1, -1, -1 };

  auto array = getMemRef(data);
  auto unrankedArray = getUnrankedMemRef(array);

  double start = 0;
  double end = 2;

  NAME_MANGLED(linspace, void, ARRAY(double), double, double)(&unrankedArray, start, end);

  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(array[i], start + i * (end - start) / (data.size() - 1));
  }
}

TEST(Runtime, log_f32)
{
  auto logFn = [](float value) -> float {
    return NAME_MANGLED(log, float, float)(value);
  };

  EXPECT_NEAR(logFn(1), 0, 0.000001);
  EXPECT_NEAR(logFn(2.718281828), 1, 0.000001);
  EXPECT_NEAR(logFn(7.389056099), 2, 0.000001);
  EXPECT_NEAR(logFn(0.367879441), -1, 0.000001);
}

TEST(Runtime, log_f64)
{
  auto logFn = [](double value) -> double {
    return NAME_MANGLED(log, double, double)(value);
  };

  EXPECT_NEAR(logFn(1), 0, 0.000001);
  EXPECT_NEAR(logFn(2.718281828), 1, 0.000001);
  EXPECT_NEAR(logFn(7.389056099), 2, 0.000001);
  EXPECT_NEAR(logFn(0.367879441), -1, 0.000001);
}

TEST(Runtime, log10_f32)
{
  auto log10Fn = [](float value) -> float {
    return NAME_MANGLED(log10, float, float)(value);
  };

  EXPECT_NEAR(log10Fn(1), 0, 0.000001);
  EXPECT_NEAR(log10Fn(10), 1, 0.000001);
  EXPECT_NEAR(log10Fn(100), 2, 0.000001);
  EXPECT_NEAR(log10Fn(0.1), -1, 0.000001);
}

TEST(Runtime, log10_f64)
{
  auto log10Fn = [](double value) -> double {
    return NAME_MANGLED(log10, double, double)(value);
  };

  EXPECT_NEAR(log10Fn(1), 0, 0.000001);
  EXPECT_NEAR(log10Fn(10), 1, 0.000001);
  EXPECT_NEAR(log10Fn(100), 2, 0.000001);
  EXPECT_NEAR(log10Fn(0.1), -1, 0.000001);
}

TEST(Runtime, maxArray_ai1)
{
  std::array<bool, 4> values = { false, true, true, false };

  auto array = getMemRef<bool, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(maxArray, bool, ARRAY(bool))(&unrankedArray);

  EXPECT_EQ(result, *std::max_element(values.begin(), values.end()));
}

TEST(Runtime, maxArray_ai32)
{
  std::array<int32_t, 4> values = { 5, 0, -3, 2 };

  auto array = getMemRef<int32_t, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(maxArray, int32_t, ARRAY(int32_t))(&unrankedArray);

  EXPECT_EQ(result, *std::max_element(values.begin(), values.end()));
}

TEST(Runtime, maxArray_ai64)
{
  std::array<int64_t, 4> values = { 5, 0, -3, 2 };

  auto array = getMemRef<int64_t, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(maxArray, int64_t, ARRAY(int64_t))(&unrankedArray);

  EXPECT_EQ(result, *std::max_element(values.begin(), values.end()));
}

TEST(Runtime, maxArray_af32)
{
  std::array<float, 4> values = { 5, 0, -3, 2 };

  auto array = getMemRef<float, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(maxArray, float, ARRAY(float))(&unrankedArray);

  EXPECT_FLOAT_EQ(result, *std::max_element(values.begin(), values.end()));
}

TEST(Runtime, maxArray_af64)
{
  std::array<double, 4> values = { 5, 0, -3, 2 };

  auto array = getMemRef<double, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(maxArray, double, ARRAY(double))(&unrankedArray);

  EXPECT_DOUBLE_EQ(result, *std::max_element(values.begin(), values.end()));
}

TEST(Runtime, maxScalars_i1_i1)
{
  std::array<bool, 4> x = { false, false, true, true };
  std::array<bool, 4> y = { false, true, false, true };

  auto maxFn = [](bool x, bool y) -> bool {
    return NAME_MANGLED(maxScalars, bool, bool, bool)(x, y);
  };

  EXPECT_EQ(maxFn(x[0], y[0]), false);
  EXPECT_EQ(maxFn(x[1], y[1]), true);
  EXPECT_EQ(maxFn(x[2], y[2]), true);
  EXPECT_EQ(maxFn(x[3], y[3]), true);
}

TEST(Runtime, maxScalars_i32_i32)
{
  std::array<int32_t, 3> x = { 0, 1, 2 };
  std::array<int32_t, 4> y = { 0, 2, 1 };

  auto maxFn = [](int32_t x, int32_t y) -> int32_t {
    return NAME_MANGLED(maxScalars, int32_t, int32_t, int32_t)(x, y);
  };

  EXPECT_EQ(maxFn(x[0], y[0]), 0);
  EXPECT_EQ(maxFn(x[1], y[1]), 2);
  EXPECT_EQ(maxFn(x[2], y[2]), 2);
}

TEST(Runtime, maxScalars_i64_i64)
{
  std::array<int64_t, 3> x = { 0, 1, 2 };
  std::array<int64_t, 4> y = { 0, 2, 1 };

  auto maxFn = [](int64_t x, int64_t y) -> int64_t {
    return NAME_MANGLED(maxScalars, int64_t, int64_t, int64_t)(x, y);
  };

  EXPECT_EQ(maxFn(x[0], y[0]), 0);
  EXPECT_EQ(maxFn(x[1], y[1]), 2);
  EXPECT_EQ(maxFn(x[2], y[2]), 2);
}

TEST(Runtime, maxScalars_f32_f32)
{
  std::array<float, 3> x = { 0, 1, 2 };
  std::array<float, 4> y = { 0, 2, 1 };

  auto maxFn = [](float x, float y) -> float {
    return NAME_MANGLED(maxScalars, float, float, float)(x, y);
  };

  EXPECT_FLOAT_EQ(maxFn(x[0], y[0]), 0);
  EXPECT_FLOAT_EQ(maxFn(x[1], y[1]), 2);
  EXPECT_FLOAT_EQ(maxFn(x[2], y[2]), 2);
}

TEST(Runtime, maxScalars_f64_f64)
{
  std::array<double, 3> x = { 0, 1, 2 };
  std::array<double, 4> y = { 0, 2, 1 };

  auto maxFn = [](double x, double y) -> double {
    return NAME_MANGLED(maxScalars, double, double, double)(x, y);
  };

  EXPECT_DOUBLE_EQ(maxFn(x[0], y[0]), 0);
  EXPECT_DOUBLE_EQ(maxFn(x[1], y[1]), 2);
  EXPECT_DOUBLE_EQ(maxFn(x[2], y[2]), 2);
}

TEST(Runtime, minArray_ai1)
{
  std::array<bool, 4> values = { false, true, true, false };

  auto array = getMemRef<bool, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(minArray, bool, ARRAY(bool))(&unrankedArray);

  EXPECT_EQ(result, *std::min_element(values.begin(), values.end()));
}

TEST(Runtime, minArray_ai32)
{
  std::array<int32_t, 4> values = { 5, 0, -3, 2 };

  auto array = getMemRef<int32_t, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(minArray, int32_t, ARRAY(int32_t))(&unrankedArray);

  EXPECT_EQ(result, *std::min_element(values.begin(), values.end()));
}

TEST(Runtime, minArray_ai64)
{
  std::array<int64_t, 4> values = { 5, 0, -3, 2 };

  auto array = getMemRef<int64_t, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(minArray, int64_t, ARRAY(int64_t))(&unrankedArray);

  EXPECT_EQ(result, *std::min_element(values.begin(), values.end()));
}

TEST(Runtime, minArray_af32)
{
  std::array<float, 4> values = { 5, 0, -3, 2 };

  auto array = getMemRef<float, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(minArray, float, ARRAY(float))(&unrankedArray);

  EXPECT_FLOAT_EQ(result, *std::min_element(values.begin(), values.end()));
}

TEST(Runtime, minArray_af64)
{
  std::array<double, 4> values = { 5, 0, -3, 2 };

  auto array = getMemRef<double, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(minArray, double, ARRAY(double))(&unrankedArray);

  EXPECT_DOUBLE_EQ(result, *std::min_element(values.begin(), values.end()));
}

TEST(Runtime, minScalars_i1_i1)
{
  std::array<bool, 4> x = { false, false, true, true };
  std::array<bool, 4> y = { false, true, false, true };

  auto minFn = [](bool x, bool y) -> bool {
    return NAME_MANGLED(minScalars, bool, bool, bool)(x, y);
  };

  EXPECT_EQ(minFn(x[0], y[0]), false);
  EXPECT_EQ(minFn(x[1], y[1]), false);
  EXPECT_EQ(minFn(x[2], y[2]), false);
  EXPECT_EQ(minFn(x[3], y[3]), true);
}

TEST(Runtime, minScalars_i32_i32)
{
  std::array<int32_t, 3> x = { 0, 1, 2 };
  std::array<int32_t, 4> y = { 0, 2, 1 };

  auto minFn = [](int32_t x, int32_t y) -> int32_t {
    return NAME_MANGLED(minScalars, int32_t, int32_t, int32_t)(x, y);
  };

  EXPECT_EQ(minFn(x[0], y[0]), 0);
  EXPECT_EQ(minFn(x[1], y[1]), 1);
  EXPECT_EQ(minFn(x[2], y[2]), 1);
}

TEST(Runtime, minScalars_i64_i64)
{
  std::array<int64_t, 3> x = { 0, 1, 2 };
  std::array<int64_t, 4> y = { 0, 2, 1 };

  auto minFn = [](int64_t x, int64_t y) -> int64_t {
    return NAME_MANGLED(minScalars, int64_t, int64_t, int64_t)(x, y);
  };

  EXPECT_EQ(minFn(x[0], y[0]), 0);
  EXPECT_EQ(minFn(x[1], y[1]), 1);
  EXPECT_EQ(minFn(x[2], y[2]), 1);
}

TEST(Runtime, minScalars_f32_f32)
{
  std::array<float, 3> x = { 0, 1, 2 };
  std::array<float, 4> y = { 0, 2, 1 };

  auto minFn = [](float x, float y) -> float {
    return NAME_MANGLED(minScalars, float, float, float)(x, y);
  };

  EXPECT_FLOAT_EQ(minFn(x[0], y[0]), 0);
  EXPECT_FLOAT_EQ(minFn(x[1], y[1]), 1);
  EXPECT_FLOAT_EQ(minFn(x[2], y[2]), 1);
}

TEST(Runtime, minScalars_f64_f64)
{
  std::array<double, 3> x = { 0, 1, 2 };
  std::array<double, 4> y = { 0, 2, 1 };

  auto minFn = [](double x, double y) -> double {
    return NAME_MANGLED(minScalars, double, double, double)(x, y);
  };

  EXPECT_DOUBLE_EQ(minFn(x[0], y[0]), 0);
  EXPECT_DOUBLE_EQ(minFn(x[1], y[1]), 1);
  EXPECT_DOUBLE_EQ(minFn(x[2], y[2]), 1);
}

TEST(Runtime, mod_i1_i1)
{
  auto modFn = [](bool x, bool y) -> bool {
    return NAME_MANGLED(mod, bool, bool, bool)(x, y);
  };

  EXPECT_EQ(modFn(false, true), false);
  EXPECT_EQ(modFn(true, true), false);
}

TEST(Runtime, mod_i32_i32)
{
  auto modFn = [](int32_t x, int32_t y) -> int32_t {
    return NAME_MANGLED(mod, int32_t, int32_t, int32_t)(x, y);
  };

  EXPECT_EQ(modFn(6, 3), 0);
  EXPECT_EQ(modFn(8, 3), 2);
  EXPECT_EQ(modFn(10, -3), -2);
  EXPECT_EQ(modFn(-10, 3), 2);
}

TEST(Runtime, mod_i64_i64)
{
  auto modFn = [](int64_t x, int64_t y) -> int64_t {
    return NAME_MANGLED(mod, int64_t, int64_t, int64_t)(x, y);
  };

  EXPECT_EQ(modFn(6, 3), 0);
  EXPECT_EQ(modFn(8, 3), 2);
  EXPECT_EQ(modFn(10, -3), -2);
  EXPECT_EQ(modFn(-10, 3), 2);
}

TEST(Runtime, mod_f32_f32)
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

TEST(Runtime, mod_f64_f64)
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

TEST(Runtime, ones_i1)
{
  std::array<bool, 4> values = { false, false, false, false };

  auto array = getMemRef<bool, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(ones, void, ARRAY(bool))(&unrankedArray);

  EXPECT_EQ(array[0][0], true);
  EXPECT_EQ(array[0][1], true);
  EXPECT_EQ(array[1][0], true);
  EXPECT_EQ(array[1][1], true);
}

TEST(Runtime, ones_i32)
{
  std::array<int32_t, 4> values = { 0, 0, 0, 0 };

  auto array = getMemRef<int32_t, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(ones, void, ARRAY(int32_t))(&unrankedArray);

  EXPECT_EQ(array[0][0], 1);
  EXPECT_EQ(array[0][1], 1);
  EXPECT_EQ(array[1][0], 1);
  EXPECT_EQ(array[1][1], 1);
}

TEST(Runtime, ones_i64)
{
  std::array<int64_t, 4> values = { 0, 0, 0, 0 };

  auto array = getMemRef<int64_t, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(ones, void, ARRAY(int64_t))(&unrankedArray);

  EXPECT_EQ(array[0][0], 1);
  EXPECT_EQ(array[0][1], 1);
  EXPECT_EQ(array[1][0], 1);
  EXPECT_EQ(array[1][1], 1);
}

TEST(Runtime, ones_f32)
{
  std::array<float, 4> values = { 0, 0, 0, 0 };

  auto array = getMemRef<float, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(ones, void, ARRAY(float))(&unrankedArray);

  EXPECT_FLOAT_EQ(array[0][0], 1);
  EXPECT_FLOAT_EQ(array[0][1], 1);
  EXPECT_FLOAT_EQ(array[1][0], 1);
  EXPECT_FLOAT_EQ(array[1][1], 1);
}

TEST(Runtime, ones_f64)
{
  std::array<double, 4> values = { 0, 0, 0, 0 };

  auto array = getMemRef<double, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(ones, void, ARRAY(double))(&unrankedArray);

  EXPECT_DOUBLE_EQ(array[0][0], 1);
  EXPECT_DOUBLE_EQ(array[0][1], 1);
  EXPECT_DOUBLE_EQ(array[1][0], 1);
  EXPECT_DOUBLE_EQ(array[1][1], 1);
}

TEST(Runtime, product_ai1)
{
  std::array<bool, 3> values = { false, true, true };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(product, bool, ARRAY(bool))(&unrankedArray);

  EXPECT_EQ(result, static_cast<bool>(std::accumulate(values.begin(), values.end(), 1, std::multiplies<>())));
}

TEST(Runtime, product_ai32)
{
  std::array<int32_t, 3> values = { 1, 2, 3 };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(product, int32_t, ARRAY(int32_t))(&unrankedArray);

  EXPECT_EQ(result, static_cast<int32_t>(std::accumulate(values.begin(), values.end(), 1, std::multiplies<>())));
}

TEST(Runtime, product_ai64)
{
  std::array<int64_t, 3> values = { 1, 2, 3 };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(product, int64_t, ARRAY(int64_t))(&unrankedArray);

  EXPECT_EQ(result, static_cast<int64_t>(std::accumulate(values.begin(), values.end(), 1, std::multiplies<>())));
}

TEST(Runtime, product_af32)
{
  std::array<float, 3> values = { 1, 2, 3 };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(product, float, ARRAY(float))(&unrankedArray);

  EXPECT_FLOAT_EQ(result, static_cast<float>(std::accumulate(values.begin(), values.end(), 1, std::multiplies<>())));
}

TEST(Runtime, product_af64)
{
  std::array<double, 3> values = { 1, 2, 3 };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(product, double, ARRAY(double))(&unrankedArray);

  EXPECT_DOUBLE_EQ(result, static_cast<double>(std::accumulate(values.begin(), values.end(), 1, std::multiplies<>())));
}

TEST(Runtime, rem_i1_i1)
{
  auto modFn = [](bool x, bool y) -> bool {
    return NAME_MANGLED(rem, bool, bool, bool)(x, y);
  };

  EXPECT_EQ(modFn(false, true), false);
  EXPECT_EQ(modFn(true, true), false);
}

TEST(Runtime, rem_i32_i32)
{
  auto modFn = [](int32_t x, int32_t y) -> int32_t {
    return NAME_MANGLED(rem, int32_t, int32_t, int32_t)(x, y);
  };

  EXPECT_EQ(modFn(6, 3), 0);
  EXPECT_EQ(modFn(8, 3), 2);
  EXPECT_EQ(modFn(10, -3), 1);
  EXPECT_EQ(modFn(-10, 3), -1);
}

TEST(Runtime, rem_i64_i64)
{
  auto modFn = [](int64_t x, int64_t y) -> int64_t {
    return NAME_MANGLED(rem, int64_t, int64_t, int64_t)(x, y);
  };

  EXPECT_EQ(modFn(6, 3), 0);
  EXPECT_EQ(modFn(8, 3), 2);
  EXPECT_EQ(modFn(10, -3), 1);
  EXPECT_EQ(modFn(-10, 3), -1);
}

TEST(Runtime, rem_f32_f32)
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

TEST(Runtime, rem_f64_f64)
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

TEST(Runtime, sign_i1)
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

TEST(Runtime, sign_i32)
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

TEST(Runtime, sign_i64)
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

TEST(Runtime, sign_f32)
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

TEST(Runtime, sign_f64)
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

TEST(Runtime, sin_f32)
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

TEST(Runtime, sin_f64)
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

TEST(Runtime, sqrt_f32)
{
  auto sqrtFn = [](float value) -> float {
    return NAME_MANGLED(sqrt, float, float)(value);
  };

  EXPECT_FLOAT_EQ(sqrtFn(0), 0);
  EXPECT_FLOAT_EQ(sqrtFn(1), 1);
  EXPECT_FLOAT_EQ(sqrtFn(4), 2);
}

TEST(Runtime, sqrt_f64)
{
  auto sqrtFn = [](double value) -> double {
    return NAME_MANGLED(sqrt, double, double)(value);
  };

  EXPECT_DOUBLE_EQ(sqrtFn(0), 0);
  EXPECT_DOUBLE_EQ(sqrtFn(1), 1);
  EXPECT_DOUBLE_EQ(sqrtFn(4), 2);
}

TEST(Runtime, sinh_f32)
{
  auto sinhFn = [](float value) -> float {
    return NAME_MANGLED(sinh, float, float)(value);
  };

  EXPECT_NEAR(sinhFn(0), 0, 0.000001);
  EXPECT_NEAR(sinhFn(1), 1.175201193, 0.000001);
}

TEST(Runtime, sinh_f64)
{
  auto sinhFn = [](double value) -> double {
    return NAME_MANGLED(sinh, double, double)(value);
  };

  EXPECT_NEAR(sinhFn(0), 0, 0.000001);
  EXPECT_NEAR(sinhFn(1), 1.175201193, 0.000001);
}

TEST(Runtime, sum_ai1)
{
  std::array<bool, 3> values = { false, true, true };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(sum, bool, ARRAY(bool))(&unrankedArray);

  EXPECT_EQ(result, static_cast<bool>(std::accumulate(values.begin(), values.end(), 0, std::plus<>())));
}

TEST(Runtime, sum_ai32)
{
  std::array<int32_t, 3> values = { 1, 2, 3 };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(sum, int32_t, ARRAY(int32_t))(&unrankedArray);

  EXPECT_EQ(result, (int32_t) std::accumulate(values.begin(), values.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_ai64)
{
  std::array<int64_t, 3> values = { 1, 2, 3 };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(sum, int64_t, ARRAY(int64_t))(&unrankedArray);

  EXPECT_EQ(result, (int64_t) std::accumulate(values.begin(), values.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_af32)
{
  std::array<float, 3> values = { 1, 2, 3 };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(sum, float, ARRAY(float))(&unrankedArray);

  EXPECT_FLOAT_EQ(result, (float) std::accumulate(values.begin(), values.end(), 0, std::plus<>()));
}

TEST(Runtime, sum_af64)
{
  std::array<double, 3> values = { 1, 2, 3 };

  auto array = getMemRef(values);
  auto unrankedArray = getUnrankedMemRef(array);

  auto result = NAME_MANGLED(sum, double, ARRAY(double))(&unrankedArray);

  EXPECT_DOUBLE_EQ(result, (double) std::accumulate(values.begin(), values.end(), 0, std::plus<>()));
}

TEST(Runtime, symmetric_ai1_ai1)
{
  std::array<bool, 9> sourceValues = { true, false, true, true, false, true, true, false, true };
  std::array<bool, 9> destinationValues = { true, false, true, true, false, true, true, false, true };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<bool>(source[2][2]));
}

TEST(Runtime, symmetric_ai1_ai32)
{
  std::array<int32_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<bool, 9> destinationValues = { true, false, true, true, false, true, true, false, true };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<bool>(source[2][2]));
}

TEST(Runtime, symmetric_ai1_ai64)
{
  std::array<int64_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<bool, 9> destinationValues = { true, false, true, true, false, true, true, false, true };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<bool>(source[2][2]));
}

TEST(Runtime, symmetric_ai1_af32)
{
  std::array<float, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<bool, 9> destinationValues = { true, false, true, true, false, true, true, false, true };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<bool>(source[2][2]));
}

TEST(Runtime, symmetric_ai1_af64)
{
  std::array<double, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<bool, 9> destinationValues = { true, false, true, true, false, true, true, false, true };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(bool), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<bool>(source[2][2]));
}

TEST(Runtime, symmetric_ai32_ai1)
{
  std::array<bool, 9> sourceValues = { true, false, true, true, false, true, true, false, true };
  std::array<int32_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int32_t>(source[2][2]));
}

TEST(Runtime, symmetric_ai32_ai32)
{
  std::array<int32_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<int32_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int32_t>(source[2][2]));
}

TEST(Runtime, symmetric_ai32_ai64)
{
  std::array<int64_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<int32_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int32_t>(source[2][2]));
}

TEST(Runtime, symmetric_ai32_af32)
{
  std::array<float, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<int32_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int32_t>(source[2][2]));
}

TEST(Runtime, symmetric_ai32_af64)
{
  std::array<double, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<int32_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int32_t), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int32_t>(source[2][2]));
}

TEST(Runtime, symmetric_ai64_ai1)
{
  std::array<bool, 9> sourceValues = { true, false, true, true, false, true, true, false, true };
  std::array<int64_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int64_t>(source[2][2]));
}

TEST(Runtime, symmetric_ai64_ai32)
{
  std::array<int32_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<int64_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int64_t>(source[2][2]));
}

TEST(Runtime, symmetric_ai64_ai64)
{
  std::array<int64_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<int64_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int64_t>(source[2][2]));
}

TEST(Runtime, symmetric_ai64_af32)
{
  std::array<float, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<int64_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int64_t>(source[2][2]));
}

TEST(Runtime, symmetric_ai64_af64)
{
  std::array<double, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<int64_t, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(int64_t), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<int64_t>(source[2][2]));
}

TEST(Runtime, symmetric_af32_ai1)
{
  std::array<bool, 9> sourceValues = { true, false, true, true, false, true, true, false, true };
  std::array<float, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<float>(source[2][2]));
}

TEST(Runtime, symmetric_af32_ai32)
{
  std::array<int32_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<float, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<float>(source[2][2]));
}

TEST(Runtime, symmetric_af32_ai64)
{
  std::array<int64_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<float, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<float>(source[2][2]));
}

TEST(Runtime, symmetric_af32_af32)
{
  std::array<float, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<float, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<float>(source[2][2]));
}

TEST(Runtime, symmetric_af32_af64)
{
  std::array<double, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<float, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(float), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<float>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<float>(source[2][2]));
}

TEST(Runtime, symmetric_af64_ai1)
{
  std::array<bool, 9> sourceValues = { true, false, true, true, false, true, true, false, true };
  std::array<double, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<double>(source[2][2]));
}

TEST(Runtime, symmetric_af64_ai32)
{
  std::array<int32_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<double, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<double>(source[2][2]));
}

TEST(Runtime, symmetric_af64_ai64)
{
  std::array<int64_t, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<double, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<double>(source[2][2]));
}

TEST(Runtime, symmetric_af64_af32)
{
  std::array<float, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<double, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<double>(source[2][2]));
}

TEST(Runtime, symmetric_af64_af64)
{
  std::array<double, 9> sourceValues = { 1, 0, 1, 0, 0, 1, 1, 0, 1 };
  std::array<double, 9> destinationValues = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 3, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(symmetric, void, ARRAY(double), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[0][2], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_EQ(destination[1][2], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<double>(source[1][2]));
  EXPECT_EQ(destination[2][2], static_cast<double>(source[2][2]));
}

TEST(Runtime, tan_f32)
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

TEST(Runtime, tan_f64)
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

TEST(Runtime, tanh_f32)
{
  auto tanhFn = [](float value) -> float {
    return NAME_MANGLED(tanh, float, float)(value);
  };

  EXPECT_NEAR(tanhFn(0), 0, 0.000001);
  EXPECT_NEAR(tanhFn(1), 0.761594155, 0.000001);
}

TEST(Runtime, tanh_f64)
{
  auto tanhFn = [](double value) -> double {
    return NAME_MANGLED(tanh, double, double)(value);
  };

  EXPECT_NEAR(tanhFn(0), 0, 0.000001);
  EXPECT_NEAR(tanhFn(1), 0.761594155, 0.000001);
}

TEST(Runtime, transpose_ai1_ai1)
{
  std::array<bool, 6> sourceValues = { false, false, false, true, true, true };
  std::array<bool, 6> destinationValues = { true, false, true, false, true, false };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
}

TEST(Runtime, transpose_ai1_ai32)
{
  std::array<int32_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<bool, 6> destinationValues = { true, false, true, false, true, false };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
}

TEST(Runtime, transpose_ai1_ai64)
{
  std::array<int64_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<bool, 6> destinationValues = { true, false, true, false, true, false };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
}

TEST(Runtime, transpose_ai1_af32)
{
  std::array<float, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<bool, 6> destinationValues = { true, false, true, false, true, false };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
}

TEST(Runtime, transpose_ai1_af64)
{
  std::array<double, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<bool, 6> destinationValues = { true, false, true, false, true, false };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(bool), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<bool>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<bool>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<bool>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<bool>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<bool>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<bool>(source[1][2]));
}

TEST(Runtime, transpose_ai32_ai1)
{
  std::array<bool, 6> sourceValues = { false, false, false, true, true, true };
  std::array<int32_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
}

TEST(Runtime, transpose_ai32_ai32)
{
  std::array<int32_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<int32_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
}

TEST(Runtime, transpose_ai32_ai64)
{
  std::array<int64_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<int32_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
}

TEST(Runtime, transpose_ai32_af32)
{
  std::array<float, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<int32_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
}

TEST(Runtime, transpose_ai32_af64)
{
  std::array<double, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<int32_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int32_t), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int32_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int32_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int32_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int32_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int32_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int32_t>(source[1][2]));
}

TEST(Runtime, transpose_ai64_ai1)
{
  std::array<bool, 6> sourceValues = { false, false, false, true, true, true };
  std::array<int64_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
}

TEST(Runtime, transpose_ai64_ai32)
{
  std::array<int32_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<int64_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
}

TEST(Runtime, transpose_ai64_ai64)
{
  std::array<int64_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<int64_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
}

TEST(Runtime, transpose_ai64_af32)
{
  std::array<float, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<int64_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
}

TEST(Runtime, transpose_ai64_af64)
{
  std::array<double, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<int64_t, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(int64_t), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], static_cast<int64_t>(source[0][0]));
  EXPECT_EQ(destination[0][1], static_cast<int64_t>(source[1][0]));
  EXPECT_EQ(destination[1][0], static_cast<int64_t>(source[0][1]));
  EXPECT_EQ(destination[1][1], static_cast<int64_t>(source[1][1]));
  EXPECT_EQ(destination[2][0], static_cast<int64_t>(source[0][2]));
  EXPECT_EQ(destination[2][1], static_cast<int64_t>(source[1][2]));
}

TEST(Runtime, transpose_af32_ai1)
{
  std::array<bool, 6> sourceValues = { false, false, false, true, true, true };
  std::array<float, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_FLOAT_EQ(destination[0][1], static_cast<float>(source[1][0]));
  EXPECT_FLOAT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_FLOAT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_FLOAT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_FLOAT_EQ(destination[2][1], static_cast<float>(source[1][2]));
}

TEST(Runtime, transpose_af32_ai32)
{
  std::array<int32_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<float, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_FLOAT_EQ(destination[0][1], static_cast<float>(source[1][0]));
  EXPECT_FLOAT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_FLOAT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_FLOAT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_FLOAT_EQ(destination[2][1], static_cast<float>(source[1][2]));
}

TEST(Runtime, transpose_af32_ai64)
{
  std::array<int64_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<float, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_FLOAT_EQ(destination[0][1], static_cast<float>(source[1][0]));
  EXPECT_FLOAT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_FLOAT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_FLOAT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_FLOAT_EQ(destination[2][1], static_cast<float>(source[1][2]));
}

TEST(Runtime, transpose_af32_af32)
{
  std::array<float, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<float, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_FLOAT_EQ(destination[0][1], static_cast<float>(source[1][0]));
  EXPECT_FLOAT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_FLOAT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_FLOAT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_FLOAT_EQ(destination[2][1], static_cast<float>(source[1][2]));
}

TEST(Runtime, transpose_af32_af64)
{
  std::array<double, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<float, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(float), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], static_cast<float>(source[0][0]));
  EXPECT_FLOAT_EQ(destination[0][1], static_cast<float>(source[1][0]));
  EXPECT_FLOAT_EQ(destination[1][0], static_cast<float>(source[0][1]));
  EXPECT_FLOAT_EQ(destination[1][1], static_cast<float>(source[1][1]));
  EXPECT_FLOAT_EQ(destination[2][0], static_cast<float>(source[0][2]));
  EXPECT_FLOAT_EQ(destination[2][1], static_cast<float>(source[1][2]));
}

TEST(Runtime, transpose_af64_ai1)
{
  std::array<bool, 6> sourceValues = { false, false, false, true, true, true };
  std::array<double, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_DOUBLE_EQ(destination[0][1], static_cast<double>(source[1][0]));
  EXPECT_DOUBLE_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_DOUBLE_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_DOUBLE_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_DOUBLE_EQ(destination[2][1], static_cast<double>(source[1][2]));
}

TEST(Runtime, transpose_af64_ai32)
{
  std::array<int32_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<double, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_DOUBLE_EQ(destination[0][1], static_cast<double>(source[1][0]));
  EXPECT_DOUBLE_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_DOUBLE_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_DOUBLE_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_DOUBLE_EQ(destination[2][1], static_cast<double>(source[1][2]));
}

TEST(Runtime, transpose_af64_ai64)
{
  std::array<int64_t, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<double, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

	NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_DOUBLE_EQ(destination[0][1], static_cast<double>(source[1][0]));
  EXPECT_DOUBLE_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_DOUBLE_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_DOUBLE_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_DOUBLE_EQ(destination[2][1], static_cast<double>(source[1][2]));
}

TEST(Runtime, transpose_af64_af32)
{
  std::array<float, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<double, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_DOUBLE_EQ(destination[0][1], static_cast<double>(source[1][0]));
  EXPECT_DOUBLE_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_DOUBLE_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_DOUBLE_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_DOUBLE_EQ(destination[2][1], static_cast<double>(source[1][2]));
}

TEST(Runtime, transpose_af64_af64)
{
  std::array<double, 6> sourceValues = { 0, 0, 0, 1, 1, 1 };
  std::array<double, 6> destinationValues = { 1, 0, 1, 0, 1, 0 };

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 3, 2 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  NAME_MANGLED(transpose, void, ARRAY(double), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], static_cast<double>(source[0][0]));
  EXPECT_DOUBLE_EQ(destination[0][1], static_cast<double>(source[1][0]));
  EXPECT_DOUBLE_EQ(destination[1][0], static_cast<double>(source[0][1]));
  EXPECT_DOUBLE_EQ(destination[1][1], static_cast<double>(source[1][1]));
  EXPECT_DOUBLE_EQ(destination[2][0], static_cast<double>(source[0][2]));
  EXPECT_DOUBLE_EQ(destination[2][1], static_cast<double>(source[1][2]));
}

TEST(Runtime, zeros_i1)
{
  std::array<bool, 4> values = { true, true, true, true };

  auto array = getMemRef<bool, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(zeros, void, ARRAY(bool))(&unrankedArray);

  EXPECT_EQ(array[0][0], false);
  EXPECT_EQ(array[0][1], false);
  EXPECT_EQ(array[1][0], false);
  EXPECT_EQ(array[1][1], false);
}

TEST(Runtime, zeros_i32)
{
  std::array<int32_t, 4> values = { 1, 1, 1, 1 };

  auto array = getMemRef<int32_t, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(zeros, void, ARRAY(int32_t))(&unrankedArray);

  EXPECT_EQ(array[0][0], 0);
  EXPECT_EQ(array[0][1], 0);
  EXPECT_EQ(array[1][0], 0);
  EXPECT_EQ(array[1][1], 0);
}

TEST(Runtime, zeros_i64)
{
  std::array<int64_t, 4> values = { 1, 1, 1, 1 };

  auto array = getMemRef<int64_t, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(zeros, void, ARRAY(int64_t))(&unrankedArray);

  EXPECT_EQ(array[0][0], 0);
  EXPECT_EQ(array[0][1], 0);
  EXPECT_EQ(array[1][0], 0);
  EXPECT_EQ(array[1][1], 0);
}

TEST(Runtime, zeros_f32)
{
  std::array<float, 4> values = { 1, 1, 1, 1 };

  auto array = getMemRef<float, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(zeros, void, ARRAY(float))(&unrankedArray);

  EXPECT_FLOAT_EQ(array[0][0], 0);
  EXPECT_FLOAT_EQ(array[0][1], 0);
  EXPECT_FLOAT_EQ(array[1][0], 0);
  EXPECT_FLOAT_EQ(array[1][1], 0);
}

TEST(Runtime, zeros_f64)
{
  std::array<double, 4> values = { 1, 1, 1, 1 };

  auto array = getMemRef<double, 2>(values.data(), { 2, 2 });
  auto unrankedArray = getUnrankedMemRef(array);

  NAME_MANGLED(zeros, void, ARRAY(double))(&unrankedArray);

  EXPECT_DOUBLE_EQ(array[0][0], 0);
  EXPECT_DOUBLE_EQ(array[0][1], 0);
  EXPECT_DOUBLE_EQ(array[1][0], 0);
  EXPECT_DOUBLE_EQ(array[1][1], 0);
}
