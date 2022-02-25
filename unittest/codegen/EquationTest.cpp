#include "gmock/gmock.h"
#include "llvm/ADT/STLExtras.h"

#include "Utils.h"

using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

using namespace ::marco::codegen::test;

// Real x;
// Real y;
//
// x = y
TEST(ScalarEquation, iterationRanges)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);

  llvm::SmallVector<mlir::Type, 2> types;
  types.push_back(RealType::get(builder.getContext()));
  types.push_back(RealType::get(builder.getContext()));

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  builder.setInsertionPointToStart(&model.body().front());
  auto equationOp = builder.create<EquationOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(equationOp.body());

  mlir::Value loadX = builder.create<LoadOp>(builder.getUnknownLoc(), model.body().getArgument(0));
  mlir::Value loadY = builder.create<LoadOp>(builder.getUnknownLoc(), model.body().getArgument(0));
  builder.create<EquationSidesOp>(builder.getUnknownLoc(), loadX, loadY);

  auto equation = Equation::build(equationOp, variables);
  EXPECT_EQ(equation->getNumOfIterationVars(), 1);

  auto ranges = equation->getIterationRanges();
  EXPECT_EQ(ranges.rank(), 1);
  EXPECT_EQ(ranges[0].getBegin(), 0);
  EXPECT_EQ(ranges[0].getEnd(), 1);
}

// Real x;
// Real y;
//
// x = y
TEST(ScalarEquation, accesses)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);

  llvm::SmallVector<mlir::Type, 2> types;
  types.push_back(RealType::get(builder.getContext()));
  types.push_back(RealType::get(builder.getContext()));

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  builder.setInsertionPointToStart(&model.body().front());
  auto equationOp = builder.create<EquationOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(equationOp.body());

  mlir::Value loadX = builder.create<LoadOp>(builder.getUnknownLoc(), model.body().getArgument(0));
  mlir::Value loadY = builder.create<LoadOp>(builder.getUnknownLoc(), model.body().getArgument(1));
  builder.create<EquationSidesOp>(builder.getUnknownLoc(), loadX, loadY);

  auto equation = Equation::build(equationOp, variables);
  auto accesses = equation->getAccesses();

  auto access1 = AccessMatcher(
      model.body().getArgument(0),
      AccessFunction(DimensionAccess::constant(0)),
      EquationPath(EquationPath::LEFT));

  auto access2 = AccessMatcher(
      model.body().getArgument(1),
      AccessFunction(DimensionAccess::constant(0)),
      EquationPath(EquationPath::RIGHT));

  EXPECT_THAT(accesses, testing::UnorderedElementsAre(access1, access2));
}

// Real[3,6] x;
// Real[3,6] y;
//
// for i in 1:2 loop
//   for j in 1:5 loop
//     x[i, j] = y[i, j];
//   end for;
// end for;
TEST(LoopEquation, explicitLoops_iterationRanges)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type, 2> types;
  types.push_back(ArrayType::get(builder.getContext(), BufferAllocationScope::heap, RealType::get(builder.getContext()), { 3, 6 }));
  types.push_back(ArrayType::get(builder.getContext(), BufferAllocationScope::heap, RealType::get(builder.getContext()), { 3, 6 }));

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  builder.setInsertionPointToStart(&model.body().front());

  auto outerLoop = builder.create<ForEquationOp>(loc, 1, 2);
  builder.setInsertionPointToStart(outerLoop.body());

  auto innerLoop = builder.create<ForEquationOp>(loc, 1, 5);
  builder.setInsertionPointToStart(innerLoop.body());

  auto equationOp = builder.create<EquationOp>(loc);
  builder.setInsertionPointToStart(equationOp.body());

  llvm::SmallVector<mlir::Value, 3> xIndexes;
  xIndexes.push_back(outerLoop.induction());
  xIndexes.push_back(innerLoop.induction());

  llvm::SmallVector<mlir::Value, 3> yIndexes;
  yIndexes.push_back(outerLoop.induction());
  yIndexes.push_back(innerLoop.induction());

  mlir::Value lhs = builder.create<LoadOp>(loc, model.body().getArgument(0), xIndexes);
  mlir::Value rhs = builder.create<LoadOp>(loc, model.body().getArgument(1), yIndexes);

  builder.create<EquationSidesOp>(loc, lhs, rhs);

  auto equation = Equation::build(equationOp, variables);

  EXPECT_EQ(equation->getNumOfIterationVars(), 2);

  auto ranges = equation->getIterationRanges();
  EXPECT_EQ(ranges.rank(), 2);

  EXPECT_EQ(ranges[0].getBegin(), 1);
  EXPECT_EQ(ranges[0].getEnd(), 3);

  EXPECT_EQ(ranges[1].getBegin(), 1);
  EXPECT_EQ(ranges[1].getEnd(), 6);
}

// Real[4,5] x;
// Real[6] y;
//
// for i in 2:5 loop
//   for j in 2:4 loop
//     x[i - 1, j + 2] = y[i];
//   end for;
// end for;
TEST(LoopEquation, explicitLoops_accesses)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type, 2> types;
  types.push_back(ArrayType::get(builder.getContext(), BufferAllocationScope::heap, RealType::get(builder.getContext()), { 4, 5 }));
  types.push_back(ArrayType::get(builder.getContext(), BufferAllocationScope::heap, RealType::get(builder.getContext()), { 6 }));

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  builder.setInsertionPointToStart(&model.body().front());
  auto outerLoop = builder.create<ForEquationOp>(loc, 2, 5);
  builder.setInsertionPointToStart(outerLoop.body());
  auto innerLoop = builder.create<ForEquationOp>(loc, 2, 4);
  builder.setInsertionPointToStart(innerLoop.body());

  auto equationOp = builder.create<EquationOp>(loc);
  builder.setInsertionPointToStart(equationOp.body());

  mlir::Value one = builder.create<ConstantOp>(loc, IntegerAttribute::get(builder.getContext(), 1));
  mlir::Value xFirstIndex = builder.create<SubOp>(loc, builder.getIndexType(), outerLoop.induction(), one);

  mlir::Value two = builder.create<ConstantOp>(loc, IntegerAttribute::get(builder.getContext(), 2));
  mlir::Value xSecondIndex = builder.create<AddOp>(loc, builder.getIndexType(), innerLoop.induction(), two);

  llvm::SmallVector<mlir::Value, 3> xIndexes;
  xIndexes.push_back(xFirstIndex);
  xIndexes.push_back(xSecondIndex);

  mlir::Value lhs = builder.create<LoadOp>(loc, model.body().getArgument(0), xIndexes);
  mlir::Value rhs = builder.create<LoadOp>(loc, model.body().getArgument(1), outerLoop.induction());

  builder.create<EquationSidesOp>(loc, lhs, rhs);

  auto equation = Equation::build(equationOp, variables);
  auto accesses = equation->getAccesses();

  auto access1 = AccessMatcher(
      model.body().getArgument(0),
      AccessFunction({
        DimensionAccess::relative(0, -1),
        DimensionAccess::relative(1, 2)
      }),
      EquationPath(EquationPath::LEFT));

  auto access2 = AccessMatcher(
      model.body().getArgument(1),
      AccessFunction(DimensionAccess::relative(0, 0)),
      EquationPath(EquationPath::RIGHT));

  EXPECT_THAT(accesses, testing::UnorderedElementsAre(access1, access2));
}

// Real[3, 5] x;
// Real[3, 5] y;
//
// x = y;
TEST(LoopEquation, implicitLoops_iterationRanges)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type, 2> types;
  auto arrayType = ArrayType::get(builder.getContext(), BufferAllocationScope::heap, RealType::get(builder.getContext()), { 3, 5 });
  types.push_back(arrayType);
  types.push_back(arrayType);

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  builder.setInsertionPointToStart(&model.body().front());
  auto equationOp = builder.create<EquationOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(equationOp.body());

  builder.create<EquationSidesOp>(loc, model.body().getArgument(0), model.body().getArgument(1));

  auto equation = Equation::build(equationOp, variables);

  EXPECT_EQ(equation->getNumOfIterationVars(), 2);

  auto ranges = equation->getIterationRanges();
  EXPECT_EQ(ranges[0].getBegin(), 0);
  EXPECT_EQ(ranges[0].getEnd(), 3);
  EXPECT_EQ(ranges[1].getBegin(), 0);
  EXPECT_EQ(ranges[1].getEnd(), 5);
}

// Real[3, 4] x;
// Real[3, 4] y;
//
// x = y;
TEST(LoopEquation, implicitLoops_accesses)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type, 2> types;
  auto arrayType = ArrayType::get(builder.getContext(), BufferAllocationScope::heap, RealType::get(builder.getContext()), { 3, 4 });
  types.push_back(arrayType);
  types.push_back(arrayType);

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  builder.setInsertionPointToStart(&model.body().front());
  auto equationOp = builder.create<EquationOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(equationOp.body());

  builder.create<EquationSidesOp>(loc, model.body().getArgument(0), model.body().getArgument(1));

  auto equation = Equation::build(equationOp, variables);
  auto accesses = equation->getAccesses();

  auto access1 = AccessMatcher(
      model.body().getArgument(0),
      AccessFunction({
          DimensionAccess::relative(0, 0),
          DimensionAccess::relative(1, 0)
      }),
      EquationPath(EquationPath::LEFT));

  auto access2 = AccessMatcher(
      model.body().getArgument(1),
      AccessFunction({
          DimensionAccess::relative(0, 0),
          DimensionAccess::relative(1, 0)
      }),
      EquationPath(EquationPath::RIGHT));

  EXPECT_THAT(accesses, testing::UnorderedElementsAre(access1, access2));
}
