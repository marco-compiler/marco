#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "marco/Modeling/AccessFunction.h"

using namespace ::marco::modeling;

TEST(ConstantDimensionAccess, creation) {
  auto access = DimensionAccess::constant(5);
  EXPECT_TRUE(access.isConstantAccess());
  EXPECT_EQ(access.getPosition(), 5);
}

TEST(RelativeDimensionAccess, creation) {
  auto access = DimensionAccess::relative(1, -3);
  EXPECT_FALSE(access.isConstantAccess());
  EXPECT_EQ(access.getInductionVariableIndex(), 1);
  EXPECT_EQ(access.getOffset(), -3);
}

TEST(ConstantDimensionAccess, mapPoint) {
  auto access = DimensionAccess::constant(5);
  Point p({2, -5, 7});
  auto mapped = access(p);
  EXPECT_EQ(mapped, 5);
}

TEST(RelativeDimensionAccess, mapPoint) {
  auto access = DimensionAccess::relative(2, -4);
  Point p({2, -5, 7});
  auto mapped = access(p);
  EXPECT_EQ(mapped, 3);
}

TEST(ConstantDimensionAccess, mapRange) {
  auto access = DimensionAccess::constant(5);

  MultidimensionalRange range({
    Range(2, 5),
    Range(1, 4),
    Range(7, 9),
  });

  auto mapped = access(range);
  EXPECT_EQ(mapped.getBegin(), 5);
  EXPECT_EQ(mapped.getEnd(), 6);
}

TEST(RelativeDimensionAccess, mapRange) {
  auto access = DimensionAccess::relative(2, -4);

  MultidimensionalRange range({
      Range(2, 5),
      Range(1, 4),
      Range(7, 9)
  });

  auto mapped = access(range);
  EXPECT_EQ(mapped.getBegin(), 3);
  EXPECT_EQ(mapped.getEnd(), 5);
}

TEST(DimensionAccess, equality) {
  EXPECT_EQ(DimensionAccess::constant(0), DimensionAccess::constant(0));
  EXPECT_NE(DimensionAccess::constant(0), DimensionAccess::constant(1));

  EXPECT_EQ(DimensionAccess::relative(1, 3), DimensionAccess::relative(1, 3));
  EXPECT_NE(DimensionAccess::relative(1, 3), DimensionAccess::relative(1, 2));

  EXPECT_NE(DimensionAccess::constant(1), DimensionAccess::relative(1, 1));
}

TEST(AccessFunction, creation) {
  AccessFunction access({
    DimensionAccess::relative(0, 1),
    DimensionAccess::constant(3)
  });

  EXPECT_EQ(access.size(), 2);
}

TEST(AccessFunction, equality) {
  AccessFunction first({
      DimensionAccess::relative(0, 1),
      DimensionAccess::constant(3)
  });

  AccessFunction second({
      DimensionAccess::relative(0, 1),
      DimensionAccess::constant(3)
  });

  EXPECT_EQ(first, second);
}

TEST(AccessFunction, isIdentity) {
  // [i0]
  AccessFunction identity1D(DimensionAccess::relative(0, 0));
  EXPECT_TRUE(identity1D.isIdentity());

  // [i0][i1]
  AccessFunction identity2D({
      DimensionAccess::relative(0, 0),
      DimensionAccess::relative(1, 0)
  });

  EXPECT_TRUE(identity2D.isIdentity());

  // [0]
  AccessFunction constant1D(DimensionAccess::constant(0));
  EXPECT_FALSE(constant1D.isIdentity());

  // [i0][i1 + 1]
  AccessFunction relativeWithOffset({
      DimensionAccess::relative(0, 0),
      DimensionAccess::relative(1, 1)
  });

  EXPECT_FALSE(relativeWithOffset.isIdentity());

  // [i0][0]
  AccessFunction identityWithConstant({
      DimensionAccess::relative(0, 0),
      DimensionAccess::constant(0)
  });

  EXPECT_FALSE(identityWithConstant.isIdentity());
}

TEST(AccessFunction, identity) {
  AccessFunction identity1D = AccessFunction::identity(1);
  EXPECT_EQ(identity1D.size(), 1);
  EXPECT_TRUE(identity1D.isIdentity());

  AccessFunction identity2D = AccessFunction::identity(2);
  EXPECT_EQ(identity2D.size(), 2);
  EXPECT_TRUE(identity2D.isIdentity());

  AccessFunction identity3D = AccessFunction::identity(3);
  EXPECT_EQ(identity3D.size(), 3);
  EXPECT_TRUE(identity3D.isIdentity());
}

TEST(AccessFunction, mapPoint) {
  AccessFunction access({
    DimensionAccess::relative(1, 3),
    DimensionAccess::relative(0, -2)
  });

  Point p({5, 1});
  auto mapped = access.map(p);

  EXPECT_EQ(mapped[0], 4);
  EXPECT_EQ(mapped[1], 3);
}

TEST(AccessFunction, mapRange) {
  AccessFunction access({
      DimensionAccess::relative(1, 3),
      DimensionAccess::relative(0, -2)
  });

  MultidimensionalRange range({
    Range(6, 8),
    Range(2, 4)
  });

  auto mapped = access.map(range);

  EXPECT_EQ(mapped.rank(), 2);

  EXPECT_EQ(mapped[0].getBegin(), 5);
  EXPECT_EQ(mapped[0].getEnd(), 7);

  EXPECT_EQ(mapped[1].getBegin(), 4);
  EXPECT_EQ(mapped[1].getEnd(), 6);
}

TEST(AccessFunction, combine) {
  AccessFunction access1({
    DimensionAccess::relative(0, 3),
    DimensionAccess::relative(1, -2)
  });

  AccessFunction access2({
    DimensionAccess::relative(1, -1),
    DimensionAccess::relative(0, 4)
  });

  auto result = access1.combine(access2);

  EXPECT_EQ(result.size(), 2);

  EXPECT_FALSE(result[0].isConstantAccess());
  EXPECT_EQ(result[0].getInductionVariableIndex(), 1);
  EXPECT_EQ(result[0].getOffset(), -3);

  EXPECT_FALSE(result[1].isConstantAccess());
  EXPECT_EQ(result[1].getInductionVariableIndex(), 0);
  EXPECT_EQ(result[1].getOffset(), 7);
}

TEST(AccessFunction, canBeInverted)
{
  AccessFunction access({
    DimensionAccess::relative(1, -2),
    DimensionAccess::relative(0, 3)
  });

  EXPECT_TRUE(access.isInvertible());
}

TEST(AccessFunction, constantAccessCantBeInverted)
{
  AccessFunction access({
    DimensionAccess::relative(1, -2),
    DimensionAccess::constant(4)
  });

  EXPECT_FALSE(access.isInvertible());
}

TEST(AccessFunction, incompleteAccessCantBeInverted)
{
  AccessFunction access({
    DimensionAccess::relative(1, -2),
    DimensionAccess::relative(1, 3)
  });

  EXPECT_FALSE(access.isInvertible());
}

TEST(AccessFunction, inverse)
{
  AccessFunction access({
    DimensionAccess::relative(1, -2),
    DimensionAccess::relative(0, 3)
  });

  ASSERT_TRUE(access.isInvertible());
  auto inverse = access.inverse();

  EXPECT_EQ(inverse.size(), 2);

  EXPECT_FALSE(inverse[0].isConstantAccess());
  EXPECT_EQ(inverse[0].getInductionVariableIndex(), 1);
  EXPECT_EQ(inverse[0].getOffset(), -3);

  EXPECT_FALSE(inverse[1].isConstantAccess());
  EXPECT_EQ(inverse[1].getInductionVariableIndex(), 0);
  EXPECT_EQ(inverse[1].getOffset(), 2);
}
