#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "marco/modeling/AccessFunction.h"

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

  IndexSet result(Point(5));

  EXPECT_EQ(mapped, result);
}

TEST(RelativeDimensionAccess, mapRange) {
  auto access = DimensionAccess::relative(2, -4);

  MultidimensionalRange range({
      Range(2, 5),
      Range(1, 4),
      Range(7, 9)
  });

  auto mapped = access(range);

  IndexSet result(MultidimensionalRange(Range(3,5)));

  EXPECT_EQ(mapped, result);
}

TEST(RelativeToArrayDimensionAccess, mapRange) {
  auto access = DimensionAccess::relativeToArray(1, {4,1,2,5,3});

  MultidimensionalRange range({
      Range(7, 9),
      Range(1, 4),
      Range(2, 4)
  });

  auto mapped = access(range);
  // taking the elements with index = range[1] = Range(1,4) = {0,1,2}

  IndexSet result(llvm::ArrayRef<Point>({
    Point(1),
    Point(2),
    Point(4)
  }));

  EXPECT_EQ(mapped, result);
}

TEST(RelativeToArrayDimensionAccess, mapRange2) {

  // {3,4,5}[i] == i+2  , iff 0 < i < 4
  // in this case the range is [2,4) -> so the result is [4,6) 

  auto access = DimensionAccess::relativeToArray(1, {3,4,5});
	auto access2 = DimensionAccess::relative(1, 2);

  MultidimensionalRange range({
      Range(7, 9),
      Range(2, 4),
      Range(1, 4),
  });

  auto mapped = access(range);
  auto mapped2 = access2(range);

  IndexSet result(llvm::ArrayRef<Point>({
    Point(4),
    Point(5)
  }));

  EXPECT_EQ(mapped, mapped2);
  EXPECT_EQ(mapped, result);
}

TEST(AccessFunction, creation) {
  AccessFunction access({
    DimensionAccess::relative(0, 1),
    DimensionAccess::constant(3)
  });

  EXPECT_EQ(access.size(), 2);
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

  auto mapped = access.map(range)[0];

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
