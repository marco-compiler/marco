#include "marco/Modeling/AccessFunctionRotoTranslation.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

#include "Common.h"

using namespace ::marco::modeling;
using namespace ::marco::test;

TEST(AccessFunctionRotoTranslation, creation_empty) {
  mlir::MLIRContext ctx;
  auto affineMap = mlir::AffineMap::get(0, 0, {}, &ctx);
  EXPECT_FALSE(AccessFunctionRotoTranslation::canBeBuilt(affineMap));
}

TEST(AccessFunctionRotoTranslation, creation_emptyDimensions) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineConstantExpr(0, &ctx));

  auto affineMap = mlir::AffineMap::get(0, 0, expressions, &ctx);
  EXPECT_FALSE(AccessFunctionRotoTranslation::canBeBuilt(affineMap));
}

TEST(AccessFunctionRotoTranslation, creation_emptyResults) {
  mlir::MLIRContext ctx;
  auto affineMap = mlir::AffineMap::get(1, 0, {}, &ctx);
  EXPECT_FALSE(AccessFunctionRotoTranslation::canBeBuilt(affineMap));
}

TEST(AccessFunctionRotoTranslation, creation_dimensions) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineDimExpr(1, &ctx));
  expressions.push_back(mlir::getAffineDimExpr(0, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);
  EXPECT_TRUE(AccessFunctionRotoTranslation::canBeBuilt(affineMap));
}

TEST(AccessFunctionRotoTranslation, creation_dimensionsWithOffsets) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(1, 5, &ctx));
  expressions.push_back(getDimWithOffset(0, -3, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);
  EXPECT_TRUE(AccessFunctionRotoTranslation::canBeBuilt(affineMap));
}

TEST(AccessFunctionRotoTranslation, creation_mixedDimensionAccesses) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineConstantExpr(0, &ctx));
  expressions.push_back(mlir::getAffineDimExpr(1, &ctx));
  expressions.push_back(getDimWithOffset(0, -3, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_TRUE(accessFunction);
}

TEST(AccessFunctionRotoTranslation, mapPoint_offsetAccess) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(2, 3, &ctx));
  expressions.push_back(getDimWithOffset(0, -2, &ctx));

  auto affineMap = mlir::AffineMap::get(3, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  Point p({3, 7, -5});
  IndexSet mapped = accessFunction->map(p);

  EXPECT_EQ(mapped, IndexSet(Point({-2, 1})));
}

TEST(AccessFunctionRotoTranslation, mapRange_offsetAccess) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(2, -4, &ctx));
  expressions.push_back(getDimWithOffset(1, 3, &ctx));

  auto affineMap = mlir::AffineMap::get(3, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  MultidimensionalRange range({
      Range(2, 5),
      Range(1, 4),
      Range(7, 9),
  });

  MultidimensionalRange mapped = accessFunction->map(range);

  ASSERT_EQ(mapped.rank(), 2);

  EXPECT_EQ(mapped[0].getBegin(), 3);
  EXPECT_EQ(mapped[0].getEnd(), 5);

  EXPECT_EQ(mapped[1].getBegin(), 4);
  EXPECT_EQ(mapped[1].getEnd(), 7);
}

TEST(AccessFunctionRotoTranslation, isIdentity_1d) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineDimExpr(0, &ctx));

  auto affineMap = mlir::AffineMap::get(1, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentity());
}

TEST(AccessFunctionRotoTranslation, isIdentity_2d) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineDimExpr(0, &ctx));
  expressions.push_back(mlir::getAffineDimExpr(1, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentity());
}

TEST(AccessFunctionRotoTranslation, isIdentity_3d) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(mlir::getAffineDimExpr(0, &ctx));
  expressions.push_back(mlir::getAffineDimExpr(1, &ctx));
  expressions.push_back(mlir::getAffineDimExpr(2, &ctx));

  auto affineMap = mlir::AffineMap::get(3, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentity());
}

TEST(AccessFunctionRotoTranslation, isIdentityLike_1d) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(0, 3, &ctx));

  auto affineMap = mlir::AffineMap::get(1, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentityLike());
}

TEST(AccessFunctionRotoTranslation, isIdentityLike_2d) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(0, 3, &ctx));
  expressions.push_back(getDimWithOffset(1, -4, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentityLike());
}

TEST(AccessFunctionRotoTranslation, isIdentityLike_3d) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(0, 3, &ctx));
  expressions.push_back(getDimWithOffset(1, -4, &ctx));
  expressions.push_back(getDimWithOffset(2, 5, &ctx));

  auto affineMap = mlir::AffineMap::get(3, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_TRUE(accessFunction->isIdentityLike());
}

TEST(AccessFunctionRotoTranslation, canBeInverted) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(1, -2, &ctx));
  expressions.push_back(getDimWithOffset(0, 3, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_TRUE(accessFunction->isInvertible());
}

TEST(AccessFunctionRotoTranslation, constantAccessCantBeInverted) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(1, -2, &ctx));
  expressions.push_back(mlir::getAffineConstantExpr(4, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_FALSE(accessFunction->isInvertible());
}

TEST(AccessFunctionRotoTranslation, incompleteAccessCantBeInverted) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(1, -2, &ctx));
  expressions.push_back(getDimWithOffset(1, 3, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  EXPECT_FALSE(accessFunction->isInvertible());
}

TEST(AccessFunctionRotoTranslation, inverse) {
  mlir::MLIRContext ctx;

  llvm::SmallVector<mlir::AffineExpr> expressions;
  expressions.push_back(getDimWithOffset(1, -2, &ctx));
  expressions.push_back(getDimWithOffset(0, 3, &ctx));

  auto affineMap = mlir::AffineMap::get(2, 0, expressions, &ctx);

  auto accessFunction =
      std::make_unique<AccessFunctionRotoTranslation>(affineMap);

  auto inverseAccessFunction = accessFunction->inverse();

  ASSERT_TRUE(inverseAccessFunction);
  EXPECT_TRUE(inverseAccessFunction->isa<AccessFunctionRotoTranslation>());

  Point p({5, 9});
  IndexSet mapped = inverseAccessFunction->map(p);
  EXPECT_EQ(mapped, IndexSet(Point({6, 7})));
}
