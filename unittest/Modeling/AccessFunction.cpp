#include "marco/Modeling/AccessFunctionGeneric.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

#include "Common.h"

using namespace ::marco::modeling;
using namespace ::marco::test;

TEST(AccessFunction, mapPoint_constantAccess) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineConstantExpr(5, &ctx));

  auto affineMap = mlir::AffineMap::get(0, 0, expressions, &ctx);
  auto accessFunction = std::make_unique<AccessFunctionGeneric>(affineMap);

  Point p({2, -5, 7});
  IndexSet mapped = accessFunction->map(p);
  EXPECT_EQ(mapped, Point(5));
}

TEST(AccessFunction, mapPoint_offsetAccess) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(2, 3, &ctx));
  expressions.push_back(getDimWithOffset(0, -2, &ctx));

  auto affineMap = mlir::AffineMap::get(3, 0, expressions, &ctx);
  auto accessFunction = std::make_unique<AccessFunctionGeneric>(affineMap);

  Point p({3, 7, -5});
  IndexSet mapped = accessFunction->map(p);
  EXPECT_EQ(mapped, Point({-2, 1}));
}

TEST(AccessFunction, isIdentity_empty) {
  mlir::MLIRContext ctx;

  auto affineMap = mlir::AffineMap::get(0, 0, {}, &ctx);
  auto accessFunction = std::make_unique<AccessFunctionGeneric>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentity());
}

TEST(AccessFunction, isIdentity_emptyDimensions) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineConstantExpr(0, &ctx));

  auto affineMap = mlir::AffineMap::get(0, 0, expressions, &ctx);
  auto accessFunction = std::make_unique<AccessFunctionGeneric>(affineMap);

  EXPECT_FALSE(accessFunction->isIdentity());
}

TEST(AccessFunction, isIdentity_emptyResults) {
  mlir::MLIRContext ctx;

  auto affineMap = mlir::AffineMap::get(1, 0, {}, &ctx);
  auto accessFunction = std::make_unique<AccessFunctionGeneric>(affineMap);

  EXPECT_FALSE(accessFunction->isIdentity());
}

TEST(AccessFunction, isIdentity_1d) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineDimExpr(0, &ctx));

  auto affineMap = mlir::AffineMap::get(1, 0, expressions, &ctx);
  auto accessFunction = std::make_unique<AccessFunctionGeneric>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentity());
}

TEST(AccessFunction, isIdentity_2d) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineDimExpr(0, &ctx));
  expressions.push_back(mlir::getAffineDimExpr(1, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);
  auto accessFunction = std::make_unique<AccessFunctionGeneric>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentity());
}

TEST(AccessFunction, isIdentity_3d) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineDimExpr(0, &ctx));
  expressions.push_back(mlir::getAffineDimExpr(1, &ctx));
  expressions.push_back(mlir::getAffineDimExpr(2, &ctx));

  auto affineMap = mlir::AffineMap::get(3, 0, expressions, &ctx);
  auto accessFunction = std::make_unique<AccessFunctionGeneric>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentity());
}

TEST(AccessFunction, combine) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions1;
  expressions1.push_back(getDimWithOffset(0, 3, &ctx));
  expressions1.push_back(getDimWithOffset(1, -2, &ctx));

  llvm::SmallVector<mlir::AffineExpr> expressions2;
  expressions2.push_back(getDimWithOffset(1, -1, &ctx));
  expressions2.push_back(getDimWithOffset(0, 4, &ctx));

  auto affineMap1 = mlir::AffineMap::get(2, 0, expressions1, &ctx);
  auto affineMap2 = mlir::AffineMap::get(2, 0, expressions2, &ctx);

  auto accessFunction1 = std::make_unique<AccessFunctionGeneric>(affineMap1);
  auto accessFunction2 = std::make_unique<AccessFunctionGeneric>(affineMap2);

  auto combinedAccessFunction = accessFunction1->combine(*accessFunction2);
  ASSERT_TRUE(combinedAccessFunction);

  Point p({5, 9});
  IndexSet mapped = combinedAccessFunction->map(p);
  EXPECT_EQ(mapped, IndexSet(Point({6, 12})));
}
