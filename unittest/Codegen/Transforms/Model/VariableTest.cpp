#include "gtest/gtest.h"

#include "Utils.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

using namespace ::marco::codegen::test;

TEST(ScalarVariable, dimensions)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);

  auto type = RealType::get(builder.getContext());
  auto model = createModel(builder, type);
  auto variables = discoverVariables(model);

  EXPECT_EQ(variables.size(), 1);
  EXPECT_EQ(variables[0]->getRank(), 1);
  EXPECT_EQ(variables[0]->getDimensionSize(0), 1);
}

TEST(ArrayVariable, dimensions)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);

  llvm::SmallVector<int64_t, 3> shape;
  shape.push_back(3);
  shape.push_back(4);
  shape.push_back(5);

  auto baseType = RealType::get(builder.getContext());
  auto arrayType = ArrayType::get(shape, baseType);
  auto model = createModel(builder, arrayType);
  auto variables = discoverVariables(model);

  EXPECT_EQ(variables.size(), 1);
  EXPECT_EQ(variables[0]->getRank(), 3);
  EXPECT_EQ(variables[0]->getDimensionSize(0), shape[0]);
  EXPECT_EQ(variables[0]->getDimensionSize(1), shape[1]);
  EXPECT_EQ(variables[0]->getDimensionSize(2), shape[2]);
}
