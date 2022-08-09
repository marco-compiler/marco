#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Utils.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

/*
TEST(IterationSpace, squareSpaceToIndexSet)
{
  std::vector<std::pair<ContiguousRange, IterationSpace*>> subDimensions;

  subDimensions.emplace_back(
      ContiguousRange(3, 5),
      new IterationSpace(IterationRange(11, 13, 1), llvm::None));

  IterationSpace iterationSpace(
      IterationRange(3, 5, 1),
      subDimensions);

  modeling::IndexSet expected;

  expected += modeling::MultidimensionalRange({
      modeling::Range(3, 6),
      modeling::Range(11, 14)
  });

  auto actual = getIndexSet(iterationSpace);

  for (auto& subDimension : subDimensions) {
    delete subDimension.second;
  }

  EXPECT_EQ(actual, expected);
}

TEST(IterationSpace, squareSpaceWithCustomStepToIndexSet)
{
  std::vector<std::pair<ContiguousRange, IterationSpace*>> subDimensions;

  subDimensions.emplace_back(
      ContiguousRange(3, 10),
      new IterationSpace(IterationRange(11, 16, 2), llvm::None));

  IterationSpace iterationSpace(
      IterationRange(3, 10, 3),
      subDimensions);

  modeling::IndexSet expected;

  expected += modeling::Point({3, 11});
  expected += modeling::Point({3, 13});
  expected += modeling::Point({3, 15});
  expected += modeling::Point({6, 11});
  expected += modeling::Point({6, 13});
  expected += modeling::Point({6, 15});
  expected += modeling::Point({9, 11});
  expected += modeling::Point({9, 13});
  expected += modeling::Point({9, 15});

  auto actual = getIndexSet(iterationSpace);

  for (auto& subDimension : subDimensions) {
    delete subDimension.second;
  }

  EXPECT_EQ(actual, expected);
}

TEST(IterationSpace, raggedSpaceToIndexSet)
{
  std::vector<std::pair<ContiguousRange, IterationSpace*>> subDimensions;

  subDimensions.emplace_back(
      ContiguousRange(3, 4),
      new IterationSpace(IterationRange(11, 13, 1), llvm::None));

  subDimensions.emplace_back(
      ContiguousRange(5, 6),
      new IterationSpace(IterationRange(15, 17, 1), llvm::None));

  IterationSpace iterationSpace(
      IterationRange(3, 6, 1),
      subDimensions);

  modeling::IndexSet expected;

  expected += modeling::MultidimensionalRange({
      modeling::Range(3, 5),
      modeling::Range(11, 14)
  });

  expected += modeling::MultidimensionalRange({
      modeling::Range(5, 7),
      modeling::Range(15, 18)
  });

  auto actual = getIndexSet(iterationSpace);

  for (auto& subDimension : subDimensions) {
    delete subDimension.second;
  }

  EXPECT_EQ(actual, expected);
}
*/
