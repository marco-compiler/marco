#include "marco/Modeling/LocalMatchingSolutions.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace ::marco::modeling;
using namespace ::marco::modeling::internal;

// clang-format off
//
// x[3]
//
// for i in [0:3)
// 	 for j in [1,3)
// 	   f(x[i]) = 0
//
// Incidence matrix:
//       (0)  (1)  (2)
// (0,1)  1    0    0
// (0,2)  1    0    0
// (1,1)  0    1    0
// (1,2)  0    1    0
// (2,1)  0    0    1
// (2,2)  0    0    1
//
// Expected solutions:
//       (0)  (1)  (2)
// (0,1)  1    0    0
// (0,2)  0    0    0
// (1,1)  0    1    0
// (1,2)  0    0    0
// (2,1)  0    0    1
// (2,2)  0    0    0
//
//       (0)  (1)  (2)
// (0,1)  0    0    0
// (0,2)  1    0    0
// (1,1)  0    0    0
// (1,2)  0    1    0
// (2,1)  0    0    0
// (2,2)  0    0    1
//
// clang-format on

TEST(LocalMatchingSolutionsMCIM, accessFunction) {
  mlir::MLIRContext context;
  MultidimensionalRange var(Range(0, 3));
  MultidimensionalRange eq({Range(0, 3), Range(1, 3)});

  MCIM input(eq, var);

  auto accessFunction = AccessFunction::build(
      mlir::AffineMap::get(2, 0, mlir::getAffineDimExpr(0, &context)));

  input.apply(*accessFunction);
  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(input)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set({0, 1}, 0);
  m0.set({1, 1}, 1);
  m0.set({2, 1}, 2);

  MCIM m1(eq, var);
  m1.set({0, 2}, 0);
  m1.set({1, 2}, 1);
  m1.set({2, 2}, 2);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1));
}

// clang-format off
//
// x[3]
//
// for i in [0:3)
// 	 for j in [1,3)
// 	   f(x[0], x[1], x[2]) = 0
//
// Incidence matrix:
//       (0)  (1)  (2)
// (0,1)  1    0    0
// (0,2)  1    0    0
// (1,1)  0    1    0
// (1,2)  0    1    0
// (2,1)  0    0    1
// (2,2)  0    0    1
//
// Expected solutions:
//       (0)  (1)  (2)
// (0,1)  1    0    0
// (0,2)  0    0    0
// (1,1)  0    0    0
// (1,2)  0    0    0
// (2,1)  0    0    0
// (2,2)  0    0    0
//
//       (0)  (1)  (2)
// (0,1)  0    0    0
// (0,2)  1    0    0
// (1,1)  0    0    0
// (1,2)  0    0    0
// (2,1)  0    0    0
// (2,2)  0    0    0
//
//       (0)  (1)  (2)
// (0,1)  0    0    0
// (0,2)  0    0    0
// (1,1)  0    1    0
// (1,2)  0    0    0
// (2,1)  0    0    0
// (2,2)  0    0    0
//
//       (0)  (1)  (2)
// (0,1)  0    0    0
// (0,2)  0    0    0
// (1,1)  0    0    0
// (1,2)  0    1    0
// (2,1)  0    0    0
// (2,2)  0    0    0
//
//       (0)  (1)  (2)
// (0,1)  0    0    0
// (0,2)  0    0    0
// (1,1)  0    0    0
// (1,2)  0    0    0
// (2,1)  0    0    1
// (2,2)  0    0    0
//
//       (0)  (1)  (2)
// (0,1)  0    0    0
// (0,2)  0    0    0
// (1,1)  0    0    0
// (1,2)  0    0    0
// (2,1)  0    0    0
// (2,2)  0    0    1
//
// clang-format on

TEST(LocalMatchingSolutionsMCIM, explicitPoints) {
  mlir::MLIRContext context;
  MultidimensionalRange var(Range(0, 3));
  MultidimensionalRange eq({Range(0, 3), Range(1, 3)});

  MCIM input(eq, var);

  input.set({0, 1}, 0);
  input.set({0, 2}, 0);
  input.set({1, 1}, 1);
  input.set({1, 2}, 1);
  input.set({2, 1}, 2);
  input.set({2, 2}, 2);

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(input)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set({0, 1}, 0);

  MCIM m1(eq, var);
  m1.set({0, 2}, 0);

  MCIM m2(eq, var);
  m2.set({1, 1}, 1);

  MCIM m3(eq, var);
  m3.set({1, 2}, 1);

  MCIM m4(eq, var);
  m4.set({2, 1}, 2);

  MCIM m5(eq, var);
  m5.set({2, 2}, 2);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2, m3, m4, m5));
}

// clang-format off
//
// x[3]
//
// for i in [0:3)
// 	 f(x[i], x[0], x[1], x[2]) = 0
//
// Incidence matrix:
//     (0)  (1)  (2)
// (0)  1    1    0
// (1)  0    1    0
// (2)  0    1    1
//
// Expected solutions:
//     (0)  (1)  (2)
// (0)  1    0    0
// (1)  0    1    0
// (2)  0    0    1
//
//     (0)  (1)  (2)
// (0)  0    1    0
// (1)  0    0    0
// (2)  0    0    0
//
//     (0)  (1)  (2)
// (0)  0    0    0
// (1)  0    1    0
// (2)  0    0    0
//
//     (0)  (1)  (2)
// (0)  0    0    0
// (1)  0    0    0
// (2)  0    1    0
//
// clang-format on

TEST(LocalMatchingSolutionsMCIM, mixedAccessFunctionAndExplicitPoints) {
  mlir::MLIRContext context;
  MultidimensionalRange var(Range(0, 3));
  MultidimensionalRange eq(Range(0, 3));

  MCIM input(eq, var);

  auto accessFunction = AccessFunction::build(
      mlir::AffineMap::get(1, 0, mlir::getAffineDimExpr(0, &context)));

  input.set(0, 1);
  input.set(1, 1);
  input.set(2, 1);

  input.apply(*accessFunction);
  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(input)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set(0, 0);
  m0.set(1, 1);
  m0.set(2, 2);

  MCIM m1(eq, var);
  m1.set(0, 1);

  MCIM m2(eq, var);
  m2.set(1, 1);

  MCIM m3(eq, var);
  m3.set(2, 1);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2, m3));
}
