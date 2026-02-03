#include "marco/Modeling/LocalMatchingSolutionsVAF.h"
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
// 			 (0)  (1)  (2)
// (0,1)  1    0    0
// (0,2)  1    0    0
// (1,1)  0    1    0
// (1,2)  0    1    0
// (2,1)  0    0    1
// (2,2)  0    0    1
//
// Expected solutions:
// 			 (0)  (1)  (2)
// (0,1)  1    0    0
// (0,2)  0    0    0
// (1,1)  0    1    0
// (1,2)  0    0    0
// (2,1)  0    0    1
// (2,2)  0    0    0
//
// 			 (0)  (1)  (2)
// (0,1)  0    0    0
// (0,2)  1    0    0
// (1,1)  0    0    0
// (1,2)  0    1    0
// (2,1)  0    0    0
// (2,2)  0    0    1
//
// clang-format on

TEST(LocalMatchingSolutionsVAF,
     2D_underdimensionedVariable_firstInductionVariable) {
  mlir::MLIRContext context;
  MultidimensionalRange var(Range(0, 3));
  MultidimensionalRange eq({Range(0, 3), Range(1, 3)});

  llvm::SmallVector<std::unique_ptr<AccessFunction>, 1> accessFunctions;

  accessFunctions.push_back(AccessFunction::build(
      mlir::AffineMap::get(2, 0, mlir::getAffineDimExpr(0, &context))));

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(IndexSet(eq), IndexSet(var),
                                                 accessFunctions)) {
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
// 	   f(x[j]) = 0
//
// Incidence matrix:
// 			 (0)  (1)  (2)
// (0,1)  0    1    0
// (0,2)  0    0    1
// (1,1)  0    1    0
// (1,2)  0    0    1
// (2,1)  0    1    0
// (2,2)  0    0    1
//
// Expected solutions:
// 			 (0)  (1)  (2)
// (0,1)  0    1    0
// (0,2)  0    0    1
// (1,1)  0    0    0
// (1,2)  0    0    0
// (2,1)  0    0    0
// (2,2)  0    0    0
//
// 			 (0)  (1)  (2)
// (0,1)  0    0    0
// (0,2)  0    0    0
// (1,1)  0    1    0
// (1,2)  0    0    1
// (2,1)  0    0    0
// (2,2)  0    0    0
//
// 			 (0)  (1)  (2)
// (0,1)  0    0    0
// (0,2)  0    0    0
// (1,1)  0    0    0
// (1,2)  0    0    0
// (2,1)  0    1    0
// (2,2)  0    0    1
//
// clang-format on

TEST(LocalMatchingSolutionsVAF,
     2D_underdimensionedVariable_secondInductionVariable) {
  mlir::MLIRContext context;
  MultidimensionalRange var(Range(0, 3));
  MultidimensionalRange eq({Range(0, 3), Range(1, 3)});

  llvm::SmallVector<std::unique_ptr<AccessFunction>, 1> accessFunctions;

  accessFunctions.push_back(AccessFunction::build(
      mlir::AffineMap::get(2, 0, mlir::getAffineDimExpr(1, &context))));

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(IndexSet(eq), IndexSet(var),
                                                 accessFunctions)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set({0, 1}, 1);
  m0.set({0, 2}, 2);

  MCIM m1(eq, var);
  m1.set({1, 1}, 1);
  m1.set({1, 2}, 2);

  MCIM m2(eq, var);
  m2.set({2, 1}, 1);
  m2.set({2, 2}, 2);

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2));
}

// clang-format off
//
// x[3][4]
//
// for i in [0:3)
// 	 for j in [1,3)
// 	   f(x[i][j]) = 0
//
// Incidence matrix:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      1      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      1      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      0      0      0      1      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
//
// Expected solutions:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      1      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      1      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      0      0      0      1      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
//
// clang-format on

TEST(LocalMatchingSolutionsVAF, 2D_allInductionVariablesUsed) {
  mlir::MLIRContext context;
  MultidimensionalRange var({Range(0, 3), Range(0, 4)});
  MultidimensionalRange eq({Range(0, 3), Range(1, 3)});

  llvm::SmallVector<std::unique_ptr<AccessFunction>, 1> accessFunctions;

  accessFunctions.push_back(AccessFunction::build(
      mlir::AffineMap::get(2, 0,
                           {mlir::getAffineDimExpr(0, &context),
                            mlir::getAffineDimExpr(1, &context)},
                           &context)));

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(IndexSet(eq), IndexSet(var),
                                                 accessFunctions)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set({0, 1}, {0, 1});
  m0.set({0, 2}, {0, 2});
  m0.set({1, 1}, {1, 1});
  m0.set({1, 2}, {1, 2});
  m0.set({2, 1}, {2, 1});
  m0.set({2, 2}, {2, 2});

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0));
}

// clang-format off
//
// x[3][4]
//
// for i in [0:3)
// 	 for j in [1,3)
// 	   f(x[j][i]) = 0
//
// Incidence matrix:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      1      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      1      0      0      0
// (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      1      0      0
// (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
//
// Expected solutions:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      1      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      1      0      0      0
// (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      1      0      0
// (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
//
// clang-format on

TEST(LocalMatchingSolutionsVAF, 2D_invertedInductionVariables) {
  mlir::MLIRContext context;
  MultidimensionalRange var({Range(0, 3), Range(0, 4)});
  MultidimensionalRange eq({Range(0, 3), Range(1, 3)});

  llvm::SmallVector<std::unique_ptr<AccessFunction>, 1> accessFunctions;

  accessFunctions.push_back(AccessFunction::build(
      mlir::AffineMap::get(2, 0,
                           {mlir::getAffineDimExpr(1, &context),
                            mlir::getAffineDimExpr(0, &context)},
                           &context)));

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(IndexSet(eq), IndexSet(var),
                                                 accessFunctions)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set({0, 1}, {1, 0});
  m0.set({0, 2}, {2, 0});
  m0.set({1, 1}, {1, 1});
  m0.set({1, 2}, {2, 1});
  m0.set({2, 1}, {1, 2});
  m0.set({2, 2}, {2, 2});

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0));
}

// clang-format off
//
// x[3][4]
//
// for i in [0:3)
//	 for j in [1,3)
// 	   f(x[2][i]) = 0
//
// Incidence matrix:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      1      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      1      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      1      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      1      0      0      0      0      0
//
// Expected solutions:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      1      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
//
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      1      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      1      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      1      0      0      0      0      0
//
// clang-format on

TEST(LocalMatchingSolutionsVAF, 2D_oneConstantIndex) {
  mlir::MLIRContext context;
  MultidimensionalRange var({Range(0, 3), Range(0, 4)});
  MultidimensionalRange eq({Range(0, 3), Range(1, 3)});

  llvm::SmallVector<std::unique_ptr<AccessFunction>, 1> accessFunctions;

  accessFunctions.push_back(AccessFunction::build(
      mlir::AffineMap::get(2, 0,
                           {mlir::getAffineConstantExpr(1, &context),
                            mlir::getAffineDimExpr(0, &context)},
                           &context)));

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(IndexSet(eq), IndexSet(var),
                                                 accessFunctions)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set({0, 1}, {1, 0});
  m0.set({1, 1}, {1, 1});
  m0.set({2, 1}, {1, 2});

  MCIM m1(eq, var);
  m1.set({0, 2}, {1, 0});
  m1.set({1, 2}, {1, 1});
  m1.set({2, 2}, {1, 2});

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1));
}

// clang-format off
//
// x[3][4]
//
// for i in [0:3)
// 	 for j in [1,3)
// 	   f(x[1][2]) = 0
//
// Incidence matrix:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      1      0      0      0      0      0
// (1,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      1      0      0      0      0      0
//
// Expected solutions:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
//
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      1      0      0      0      0      0
// (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
//
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
//
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
//
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      1      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
//
//			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      1      0      0      0      0      0
//
// clang-format on

TEST(LocalMatchingSolutionsVAF, 2D_allConstantIndexes) {
  mlir::MLIRContext context;
  MultidimensionalRange var({Range(0, 3), Range(0, 4)});
  MultidimensionalRange eq({Range(0, 3), Range(1, 3)});

  llvm::SmallVector<std::unique_ptr<AccessFunction>, 1> accessFunctions;

  accessFunctions.push_back(AccessFunction::build(
      mlir::AffineMap::get(2, 0,
                           {mlir::getAffineConstantExpr(1, &context),
                            mlir::getAffineConstantExpr(2, &context)},
                           &context)));

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(IndexSet(eq), IndexSet(var),
                                                 accessFunctions)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set({0, 1}, {1, 2});

  MCIM m1(eq, var);
  m1.set({0, 2}, {1, 2});

  MCIM m2(eq, var);
  m2.set({1, 1}, {1, 2});

  MCIM m3(eq, var);
  m3.set({1, 2}, {1, 2});

  MCIM m4(eq, var);
  m4.set({2, 1}, {1, 2});

  MCIM m5(eq, var);
  m5.set({2, 2}, {1, 2});

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2, m3, m4, m5));
}

// clang-format off
//
// x[3][4]
//
// for i in [0:3)
// 	 for j in [1,3)
// 	   x[j][j]
//
// Incidence matrix:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      1      0
// (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      1      0
// (2,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
//
// Expected solutions:
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      1      0
// (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
//
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      1      0
// (2,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      0      0
//
// 			 (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)  (2,0)  (2,1)  (2,2)  (2,3)
// (0,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (0,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,1)   0      0      0      0      0      0      0      0      0      0      0      0
// (1,2)   0      0      0      0      0      0      0      0      0      0      0      0
// (2,1)   0      0      0      0      0      1      0      0      0      0      0      0
// (2,2)   0      0      0      0      0      0      0      0      0      0      1      0
//
// clang-format on

TEST(LocalMatchingSolutionsVAF, 2D_repeatedInductionVariable) {
  mlir::MLIRContext context;
  MultidimensionalRange var({Range(0, 3), Range(0, 4)});
  MultidimensionalRange eq({Range(0, 3), Range(1, 3)});

  llvm::SmallVector<std::unique_ptr<AccessFunction>, 1> accessFunctions;

  accessFunctions.push_back(AccessFunction::build(
      mlir::AffineMap::get(2, 0,
                           {mlir::getAffineDimExpr(1, &context),
                            mlir::getAffineDimExpr(1, &context)},
                           &context)));

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(IndexSet(eq), IndexSet(var),
                                                 accessFunctions)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set({0, 1}, {1, 1});
  m0.set({0, 2}, {2, 2});

  MCIM m1(eq, var);
  m1.set({1, 1}, {1, 1});
  m1.set({1, 2}, {2, 2});

  MCIM m2(eq, var);
  m2.set({2, 1}, {1, 1});
  m2.set({2, 2}, {2, 2});

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2));
}

// clang-format off
//
// x[3][4][5]
//
// for i in [0:3)
// 	 for j in [1,3)
// 	   for k in [1,4)
// 	     x[j][j][i]
//
// clang-format on

TEST(LocalMatchingSolutionsVAF, 3D_repeatedInductionVariable) {
  mlir::MLIRContext context;
  MultidimensionalRange var({Range(0, 3), Range(0, 4), Range(0, 5)});
  MultidimensionalRange eq({Range(0, 3), Range(1, 3), Range(1, 4)});

  llvm::SmallVector<std::unique_ptr<AccessFunction>, 1> accessFunctions;

  accessFunctions.push_back(AccessFunction::build(mlir::AffineMap::get(
      3, 0,
      {mlir::getAffineDimExpr(1, &context), mlir::getAffineDimExpr(1, &context),
       mlir::getAffineDimExpr(0, &context)},
      &context)));

  std::vector<MCIM> solutions;

  for (auto solution : solveLocalMatchingProblem(IndexSet(eq), IndexSet(var),
                                                 accessFunctions)) {
    solutions.push_back(solution);
  }

  MCIM m0(eq, var);
  m0.set({0, 1, 1}, {1, 1, 0});
  m0.set({0, 2, 1}, {2, 2, 0});
  m0.set({1, 1, 1}, {1, 1, 1});
  m0.set({1, 2, 1}, {2, 2, 1});
  m0.set({2, 1, 1}, {1, 1, 2});
  m0.set({2, 2, 1}, {2, 2, 2});

  MCIM m1(eq, var);
  m1.set({0, 1, 2}, {1, 1, 0});
  m1.set({0, 2, 2}, {2, 2, 0});
  m1.set({1, 1, 2}, {1, 1, 1});
  m1.set({1, 2, 2}, {2, 2, 1});
  m1.set({2, 1, 2}, {1, 1, 2});
  m1.set({2, 2, 2}, {2, 2, 2});

  MCIM m2(eq, var);
  m2.set({0, 1, 3}, {1, 1, 0});
  m2.set({0, 2, 3}, {2, 2, 0});
  m2.set({1, 1, 3}, {1, 1, 1});
  m2.set({1, 2, 3}, {2, 2, 1});
  m2.set({2, 1, 3}, {1, 1, 2});
  m2.set({2, 2, 3}, {2, 2, 2});

  EXPECT_THAT(solutions, testing::UnorderedElementsAre(m0, m1, m2));
}
