#include "marco/Runtime/Support/UtilityFunctions.h"
#include "gtest/gtest.h"

#include "Utils.h"

#include <iostream>

TEST(Runtime, clone_ai1_ai1)
{
  std::array<bool, 6> destinationValues = { false, false, false, false, false, false };
  std::array<bool, 6> sourceValues = { true, true, true, true, true, true };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], true);
  EXPECT_EQ(destination[0][2], true);

  EXPECT_EQ(destination[1][0], true);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], true);
}

TEST(Runtime, clone_ai1_ai32)
{
  std::array<bool, 6> destinationValues = { false, false, false, false, false, false };
  std::array<int32_t, 6> sourceValues = { 1, 1, 1, 1, 1, 1 };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], true);
  EXPECT_EQ(destination[0][2], true);

  EXPECT_EQ(destination[1][0], true);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], true);
}

TEST(Runtime, clone_ai1_ai64)
{
  std::array<bool, 6> destinationValues = { false, false, false, false, false, false };
  std::array<int64_t, 6> sourceValues = { 1, 1, 1, 1, 1, 1 };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], true);
  EXPECT_EQ(destination[0][2], true);

  EXPECT_EQ(destination[1][0], true);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], true);
}

TEST(Runtime, clone_ai1_af32)
{
  std::array<bool, 6> destinationValues = { false, false, false, false, false, false };
  std::array<float, 6> sourceValues = { 1, 1, 1, 1, 1, 1 };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], true);
  EXPECT_EQ(destination[0][2], true);

  EXPECT_EQ(destination[1][0], true);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], true);
}

TEST(Runtime, clone_ai1_af64)
{
  std::array<bool, 6> destinationValues = { false, false, false, false, false, false };
  std::array<double, 6> sourceValues = { 1, 1, 1, 1, 1, 1 };

  auto destination = getMemRef<bool, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(bool), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], true);
  EXPECT_EQ(destination[0][1], true);
  EXPECT_EQ(destination[0][2], true);

  EXPECT_EQ(destination[1][0], true);
  EXPECT_EQ(destination[1][1], true);
  EXPECT_EQ(destination[1][2], true);
}

TEST(Runtime, clone_ai32_ai1)
{
  std::array<int32_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<bool, 6> sourceValues = { true, true, true, true, true, true };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 1);
  EXPECT_EQ(destination[0][2], 1);

  EXPECT_EQ(destination[1][0], 1);
  EXPECT_EQ(destination[1][1], 1);
  EXPECT_EQ(destination[1][2], 1);
}

TEST(Runtime, clone_ai32_ai32)
{
  std::array<int32_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<int32_t, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 2);
  EXPECT_EQ(destination[0][2], 3);

  EXPECT_EQ(destination[1][0], 4);
  EXPECT_EQ(destination[1][1], 5);
  EXPECT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_ai32_ai64)
{
  std::array<int32_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<int64_t, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 2);
  EXPECT_EQ(destination[0][2], 3);

  EXPECT_EQ(destination[1][0], 4);
  EXPECT_EQ(destination[1][1], 5);
  EXPECT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_ai32_af32)
{
  std::array<int32_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<float, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 2);
  EXPECT_EQ(destination[0][2], 3);

  EXPECT_EQ(destination[1][0], 4);
  EXPECT_EQ(destination[1][1], 5);
  EXPECT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_ai32_af64)
{
  std::array<int32_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<double, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<int32_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int32_t), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 2);
  EXPECT_EQ(destination[0][2], 3);

  EXPECT_EQ(destination[1][0], 4);
  EXPECT_EQ(destination[1][1], 5);
  EXPECT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_ai64_ai1)
{
  std::array<int64_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<bool, 6> sourceValues = { true, true, true, true, true, true };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 1);
  EXPECT_EQ(destination[0][2], 1);

  EXPECT_EQ(destination[1][0], 1);
  EXPECT_EQ(destination[1][1], 1);
  EXPECT_EQ(destination[1][2], 1);
}

TEST(Runtime, clone_ai64_ai32)
{
  std::array<int64_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<int32_t, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 2);
  EXPECT_EQ(destination[0][2], 3);

  EXPECT_EQ(destination[1][0], 4);
  EXPECT_EQ(destination[1][1], 5);
  EXPECT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_ai64_ai64)
{
  std::array<int64_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<int64_t, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 2);
  EXPECT_EQ(destination[0][2], 3);

  EXPECT_EQ(destination[1][0], 4);
  EXPECT_EQ(destination[1][1], 5);
  EXPECT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_ai64_af32)
{
  std::array<int64_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<float, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 2);
  EXPECT_EQ(destination[0][2], 3);

  EXPECT_EQ(destination[1][0], 4);
  EXPECT_EQ(destination[1][1], 5);
  EXPECT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_ai64_af64)
{
  std::array<int64_t, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<double, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<int64_t, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(int64_t), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_EQ(destination[0][0], 1);
  EXPECT_EQ(destination[0][1], 2);
  EXPECT_EQ(destination[0][2], 3);

  EXPECT_EQ(destination[1][0], 4);
  EXPECT_EQ(destination[1][1], 5);
  EXPECT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_af32_ai1)
{
  std::array<float, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<bool, 6> sourceValues = { true, true, true, true, true, true };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(float), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 1);
  EXPECT_FLOAT_EQ(destination[0][2], 1);

  EXPECT_FLOAT_EQ(destination[1][0], 1);
  EXPECT_FLOAT_EQ(destination[1][1], 1);
  EXPECT_FLOAT_EQ(destination[1][2], 1);
}

TEST(Runtime, clone_af32_ai32)
{
  std::array<float, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<int32_t, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(float), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 2);
  EXPECT_FLOAT_EQ(destination[0][2], 3);

  EXPECT_FLOAT_EQ(destination[1][0], 4);
  EXPECT_FLOAT_EQ(destination[1][1], 5);
  EXPECT_FLOAT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_af32_ai64)
{
  std::array<float, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<int64_t, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(float), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 2);
  EXPECT_FLOAT_EQ(destination[0][2], 3);

  EXPECT_FLOAT_EQ(destination[1][0], 4);
  EXPECT_FLOAT_EQ(destination[1][1], 5);
  EXPECT_FLOAT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_af32_af32)
{
  std::array<float, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<float, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(float), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 2);
  EXPECT_FLOAT_EQ(destination[0][2], 3);

  EXPECT_FLOAT_EQ(destination[1][0], 4);
  EXPECT_FLOAT_EQ(destination[1][1], 5);
  EXPECT_FLOAT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_af32_af64)
{
  std::array<float, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<double, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<float, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(float), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_FLOAT_EQ(destination[0][0], 1);
  EXPECT_FLOAT_EQ(destination[0][1], 2);
  EXPECT_FLOAT_EQ(destination[0][2], 3);

  EXPECT_FLOAT_EQ(destination[1][0], 4);
  EXPECT_FLOAT_EQ(destination[1][1], 5);
  EXPECT_FLOAT_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_af64_ai1)
{
  std::array<double, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<bool, 6> sourceValues = { true, true, true, true, true, true };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<bool, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(double), ARRAY(bool))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 1);
  EXPECT_DOUBLE_EQ(destination[0][2], 1);

  EXPECT_DOUBLE_EQ(destination[1][0], 1);
  EXPECT_DOUBLE_EQ(destination[1][1], 1);
  EXPECT_DOUBLE_EQ(destination[1][2], 1);
}

TEST(Runtime, clone_af64_ai32)
{
  std::array<double, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<int32_t, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int32_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(double), ARRAY(int32_t))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 2);
  EXPECT_DOUBLE_EQ(destination[0][2], 3);

  EXPECT_DOUBLE_EQ(destination[1][0], 4);
  EXPECT_DOUBLE_EQ(destination[1][1], 5);
  EXPECT_DOUBLE_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_af64_ai64)
{
  std::array<double, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<int64_t, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<int64_t, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(double), ARRAY(int64_t))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 2);
  EXPECT_DOUBLE_EQ(destination[0][2], 3);

  EXPECT_DOUBLE_EQ(destination[1][0], 4);
  EXPECT_DOUBLE_EQ(destination[1][1], 5);
  EXPECT_DOUBLE_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_af64_af32)
{
  std::array<double, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<float, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<float, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(double), ARRAY(float))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 2);
  EXPECT_DOUBLE_EQ(destination[0][2], 3);

  EXPECT_DOUBLE_EQ(destination[1][0], 4);
  EXPECT_DOUBLE_EQ(destination[1][1], 5);
  EXPECT_DOUBLE_EQ(destination[1][2], 6);
}

TEST(Runtime, clone_af64_af64)
{
  std::array<double, 6> destinationValues = { 0, 0, 0, 0, 0, 0 };
  std::array<double, 6> sourceValues = { 1, 2, 3, 4, 5, 6 };

  auto destination = getMemRef<double, 2>(destinationValues.data(), { 2, 3 });
  auto unrankedDestination = getUnrankedMemRef(destination);

  auto source = getMemRef<double, 2>(sourceValues.data(), { 2, 3 });
  auto unrankedSource = getUnrankedMemRef(source);

  NAME_MANGLED(clone, void, ARRAY(double), ARRAY(double))(&unrankedDestination, &unrankedSource);

  EXPECT_DOUBLE_EQ(destination[0][0], 1);
  EXPECT_DOUBLE_EQ(destination[0][1], 2);
  EXPECT_DOUBLE_EQ(destination[0][2], 3);

  EXPECT_DOUBLE_EQ(destination[1][0], 4);
  EXPECT_DOUBLE_EQ(destination[1][1], 5);
  EXPECT_DOUBLE_EQ(destination[1][2], 6);
}
