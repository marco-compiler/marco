#include "marco/Runtime/Support/Math.h"
#include "gtest/gtest.h"

TEST(Runtime, pow_i1_i1_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, bool)(false, true), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, bool)(true, false), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, bool)(true, true), true);
}

TEST(Runtime, pow_i1_i1_i32)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, int32_t)(false, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, int32_t)(true, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, int32_t)(true, 1), true);
}

TEST(Runtime, pow_i1_i1_i64)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, int64_t)(false, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, int64_t)(true, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, int64_t)(true, 1), true);
}

TEST(Runtime, pow_i1_i1_f32)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, float)(false, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, float)(true, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, float)(true, 1), true);
}

TEST(Runtime, pow_i1_i1_f64)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, double)(false, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, double)(true, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, bool, double)(true, 1), true);
}

TEST(Runtime, pow_i1_i32_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, int32_t, bool)(0, true), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int32_t, bool)(1, false), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int32_t, bool)(1, true), true);
}

TEST(Runtime, pow_i1_i32_i32)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, int32_t, int32_t)(0, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int32_t, int32_t)(1, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int32_t, int32_t)(1, 1), true);
}

TEST(Runtime, pow_i1_i32_f32)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, int32_t, float)(0, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int32_t, float)(1, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int32_t, float)(1, 1), true);
}

TEST(Runtime, pow_i1_i64_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, int64_t, bool)(0, true), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int64_t, bool)(1, false), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int64_t, bool)(1, true), true);
}

TEST(Runtime, pow_i1_i64_i64)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, int64_t, int64_t)(0, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int64_t, int64_t)(1, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int64_t, int64_t)(1, 1), true);
}

TEST(Runtime, pow_i1_i64_f64)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, int64_t, double)(0, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int64_t, double)(1, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, int64_t, double)(1, 1), true);
}

TEST(Runtime, pow_i1_f32_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, float, bool)(0, true), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, float, bool)(1, false), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, float, bool)(1, true), true);
}

TEST(Runtime, pow_i1_f32_i32)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, float, int32_t)(0, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, float, int32_t)(1, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, float, int32_t)(1, 1), true);
}

TEST(Runtime, pow_i1_f32_f32)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, float, float)(0, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, float, float)(1, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, float, float)(1, 1), true);
}

TEST(Runtime, pow_i1_f64_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, double, bool)(0, true), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, double, bool)(1, false), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, double, bool)(1, true), true);
}

TEST(Runtime, pow_i1_f64_i64)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, double, int64_t)(0, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, double, int64_t)(1, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, double, int64_t)(1, 1), true);
}

TEST(Runtime, pow_i1_f64_f64)
{
  EXPECT_EQ(NAME_MANGLED(pow, bool, double, double)(0, 1), false);
  EXPECT_EQ(NAME_MANGLED(pow, bool, double, double)(1, 0), true);
  EXPECT_EQ(NAME_MANGLED(pow, bool, double, double)(1, 1), true);
}

TEST(Runtime, pow_i32_i1_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, bool, bool)(false, true), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, bool, bool)(true, false), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, bool, bool)(true, true), 1);
}

TEST(Runtime, pow_i32_i1_i32)
{
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, bool, int32_t)(false, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, bool, int32_t)(true, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, bool, int32_t)(true, 2), 1);
}

TEST(Runtime, pow_i32_i1_f32)
{
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, bool, float)(false, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, bool, float)(true, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, bool, float)(true, 2), 1);
}

TEST(Runtime, pow_i32_i32_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, int32_t, bool)(0, true), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, int32_t, bool)(3, false), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, int32_t, bool)(3, true), 3);
}

TEST(Runtime, pow_i32_i32_i32)
{
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, int32_t, int32_t)(0, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, int32_t, int32_t)(3, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, int32_t, int32_t)(3, 2), 9);
}

TEST(Runtime, pow_i32_i32_f32)
{
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, int32_t, float)(0, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, int32_t, float)(3, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, int32_t, float)(3, 2), 9);
}

TEST(Runtime, pow_i32_f32_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, float, bool)(0, true), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, float, bool)(3, false), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, float, bool)(3, true), 3);
}

TEST(Runtime, pow_i32_f32_i32)
{
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, float, int32_t)(0, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, float, int32_t)(3, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, float, int32_t)(3, 2), 9);
}

TEST(Runtime, pow_i32_f32_f32)
{
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, float, float)(0, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, float, float)(3, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int32_t, float, float)(3, 2), 9);
}

TEST(Runtime, pow_i64_i1_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, bool, bool)(false, true), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, bool, bool)(true, false), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, bool, bool)(true, true), 1);
}

TEST(Runtime, pow_i64_i1_i64)
{
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, bool, int64_t)(false, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, bool, int64_t)(true, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, bool, int64_t)(true, 2), 1);
}

TEST(Runtime, pow_i64_i1_f64)
{
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, bool, double)(false, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, bool, double)(true, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, bool, double)(true, 2), 1);
}

TEST(Runtime, pow_i64_i64_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, int64_t, bool)(0, true), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, int64_t, bool)(3, false), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, int64_t, bool)(3, true), 3);
}

TEST(Runtime, pow_i64_i64_i64)
{
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, int64_t, int64_t)(0, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, int64_t, int64_t)(3, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, int64_t, int64_t)(3, 2), 9);
}

TEST(Runtime, pow_i64_i64_f64)
{
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, int64_t, double)(0, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, int64_t, double)(3, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, int64_t, double)(3, 2), 9);
}

TEST(Runtime, pow_i64_f64_i1)
{
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, double, bool)(0, true), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, double, bool)(3, false), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, double, bool)(3, true), 3);
}

TEST(Runtime, pow_i64_f64_i64)
{
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, double, int64_t)(0, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, double, int64_t)(3, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, double, int64_t)(3, 2), 9);
}

TEST(Runtime, pow_i64_f64_f64)
{
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, double, double)(0, 2), 0);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, double, double)(3, 0), 1);
  EXPECT_EQ(NAME_MANGLED(pow, int64_t, double, double)(3, 2), 9);
}

TEST(Runtime, pow_f32_i1_i1)
{
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, bool, bool)(false, true), 0);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, bool, bool)(true, false), 1);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, bool, bool)(true, true), 1);
}

TEST(Runtime, pow_f32_i1_i32)
{
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, bool, int32_t)(false, 2), 0);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, bool, int32_t)(true, 0), 1);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, bool, int32_t)(true, 2), 1);
}

TEST(Runtime, pow_f32_i1_f32)
{
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, bool, float)(false, 2), 0);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, bool, float)(true, 0), 1);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, bool, float)(true, 2), 1);
}

TEST(Runtime, pow_f32_i32_i1)
{
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, int32_t, bool)(0, true), 0);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, int32_t, bool)(3, false), 1);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, int32_t, bool)(3, true), 3);
}

TEST(Runtime, pow_f32_i32_i32)
{
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, int32_t, int32_t)(0, 2), 0);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, int32_t, int32_t)(3, 0), 1);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, int32_t, int32_t)(3, 2), 9);
}

TEST(Runtime, pow_f32_i32_f32)
{
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, int32_t, float)(0, 2), 0);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, int32_t, float)(3, 0), 1);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, int32_t, float)(3, 2), 9);
}

TEST(Runtime, pow_f32_f32_i1)
{
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, float, bool)(0, true), 0);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, float, bool)(3, false), 1);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, float, bool)(3, true), 3);
}

TEST(Runtime, pow_f32_f32_i32)
{
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, float, int32_t)(0, 2), 0);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, float, int32_t)(3, 0), 1);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, float, int32_t)(3, 2), 9);
}

TEST(Runtime, pow_f32_f32_f32)
{
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, float, float)(0, 2), 0);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, float, float)(3, 0), 1);
  EXPECT_FLOAT_EQ(NAME_MANGLED(pow, float, float, float)(3, 2), 9);
}

TEST(Runtime, pow_f64_i1_i1)
{
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, bool, bool)(false, true), 0);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, bool, bool)(true, false), 1);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, bool, bool)(true, true), 1);
}

TEST(Runtime, pow_f64_i1_i64)
{
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, bool, int64_t)(false, 2), 0);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, bool, int64_t)(true, 0), 1);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, bool, int64_t)(true, 2), 1);
}

TEST(Runtime, pow_f64_i1_f64)
{
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, bool, double)(false, 2), 0);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, bool, double)(true, 0), 1);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, bool, double)(true, 2), 1);
}

TEST(Runtime, pow_f64_i64_i1)
{
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, int64_t, bool)(0, true), 0);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, int64_t, bool)(3, false), 1);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, int64_t, bool)(3, true), 3);
}

TEST(Runtime, pow_f64_i64_i64)
{
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, int64_t, int64_t)(0, 2), 0);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, int64_t, int64_t)(3, 0), 1);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, int64_t, int64_t)(3, 2), 9);
}

TEST(Runtime, pow_f64_i64_f64)
{
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, int64_t, double)(0, 2), 0);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, int64_t, double)(3, 0), 1);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, int64_t, double)(3, 2), 9);
}

TEST(Runtime, pow_f64_f64_i1)
{
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, double, bool)(0, true), 0);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, double, bool)(3, false), 1);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, double, bool)(3, true), 3);
}

TEST(Runtime, pow_f64_f64_i64)
{
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, double, int64_t)(0, 2), 0);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, double, int64_t)(3, 0), 1);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, double, int64_t)(3, 2), 9);
}

TEST(Runtime, pow_f64_f64_f64)
{
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, double, double)(0, 2), 0);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, double, double)(3, 0), 1);
  EXPECT_DOUBLE_EQ(NAME_MANGLED(pow, double, double, double)(3, 2), 9);
}
