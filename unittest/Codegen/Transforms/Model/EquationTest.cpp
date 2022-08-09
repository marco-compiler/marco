#include "gmock/gmock.h"
#include "llvm/ADT/STLExtras.h"

#include "Utils.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

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

  auto equationOp = createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange indices) {
    mlir::Value loadX = nested.create<LoadOp>(builder.getUnknownLoc(), model.getBodyRegion().getArgument(0));
    mlir::Value loadY = nested.create<LoadOp>(builder.getUnknownLoc(), model.getBodyRegion().getArgument(1));
    createEquationSides(nested, loadX, loadY);
  });

  auto equation = Equation::build(equationOp, variables);
  EXPECT_EQ(equation->getNumOfIterationVars(), 1);

  auto iterationRanges = equation->getIterationRanges();
  assert(iterationRanges.isSingleMultidimensionalRange());
  auto ranges = equation->getIterationRanges().minContainingRange();
  
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

  auto equationOp = createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange indices) {
    mlir::Value loadX = nested.create<LoadOp>(builder.getUnknownLoc(), model.getBodyRegion().getArgument(0));
    mlir::Value loadY = nested.create<LoadOp>(builder.getUnknownLoc(), model.getBodyRegion().getArgument(1));
    createEquationSides(nested, loadX, loadY);
  });

  auto equation = Equation::build(equationOp, variables);
  auto accesses = equation->getAccesses();

  auto access1 = AccessMatcher(
      model.getBodyRegion().getArgument(0),
      AccessFunction(DimensionAccess::constant(0)),
      EquationPath(EquationPath::LEFT));

  auto access2 = AccessMatcher(
      model.getBodyRegion().getArgument(1),
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
  types.push_back(ArrayType::get({ 3, 6 }, RealType::get(builder.getContext())));
  types.push_back(ArrayType::get({ 3, 6 }, RealType::get(builder.getContext())));

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  std::vector<std::pair<long, long>> equationRanges;
  equationRanges.emplace_back(1, 2);
  equationRanges.emplace_back(1, 5);

  auto equationOp = createEquation(builder, model, equationRanges, [&](mlir::OpBuilder& nested, mlir::ValueRange indices) {
    llvm::SmallVector<mlir::Value, 3> xIndexes;
    xIndexes.push_back(indices[0]);
    xIndexes.push_back(indices[1]);

    llvm::SmallVector<mlir::Value, 3> yIndexes;
    yIndexes.push_back(indices[0]);
    yIndexes.push_back(indices[1]);

    mlir::Value lhs = nested.create<LoadOp>(loc, model.getBodyRegion().getArgument(0), xIndexes);
    mlir::Value rhs = nested.create<LoadOp>(loc, model.getBodyRegion().getArgument(1), yIndexes);

    createEquationSides(nested, lhs, rhs);
  });

  auto equation = Equation::build(equationOp, variables);

  EXPECT_EQ(equation->getNumOfIterationVars(), 2);

  auto iterationRanges = equation->getIterationRanges();
  assert(iterationRanges.isSingleMultidimensionalRange());
  auto ranges = equation->getIterationRanges().minContainingRange();

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
  types.push_back(ArrayType::get({ 4, 5 }, RealType::get(builder.getContext())));
  types.push_back(ArrayType::get({ 6 }, RealType::get(builder.getContext())));

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  std::vector<std::pair<long, long>> equationRanges;
  equationRanges.emplace_back(2, 5);
  equationRanges.emplace_back(2, 4);

  auto equationOp = createEquation(builder, model, equationRanges, [&](mlir::OpBuilder& nested, mlir::ValueRange indices) {
    mlir::Value one = nested.create<ConstantOp>(loc, IntegerAttr::get(builder.getContext(), 1));
    mlir::Value xFirstIndex = nested.create<SubOp>(loc, builder.getIndexType(), indices[0], one);

    mlir::Value two = nested.create<ConstantOp>(loc, IntegerAttr::get(builder.getContext(), 2));
    mlir::Value xSecondIndex = nested.create<AddOp>(loc, builder.getIndexType(), indices[1], two);

    llvm::SmallVector<mlir::Value, 3> xIndexes;
    xIndexes.push_back(xFirstIndex);
    xIndexes.push_back(xSecondIndex);

    mlir::Value lhs = nested.create<LoadOp>(loc, model.getBodyRegion().getArgument(0), xIndexes);
    mlir::Value rhs = nested.create<LoadOp>(loc, model.getBodyRegion().getArgument(1), indices[0]);

    createEquationSides(nested, lhs, rhs);
  });

  auto equation = Equation::build(equationOp, variables);
  auto accesses = equation->getAccesses();

  auto access1 = AccessMatcher(
      model.getBodyRegion().getArgument(0),
      AccessFunction({
        DimensionAccess::relative(0, -1),
        DimensionAccess::relative(1, 2)
      }),
      EquationPath(EquationPath::LEFT));

  auto access2 = AccessMatcher(
      model.getBodyRegion().getArgument(1),
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

  llvm::SmallVector<mlir::Type, 2> types;
  auto arrayType = ArrayType::get({ 3, 5 }, RealType::get(builder.getContext()));
  types.push_back(arrayType);
  types.push_back(arrayType);

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  auto equationOp = createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange indices) {
    mlir::Value lhs = model.getBodyRegion().getArgument(0);
    mlir::Value rhs = model.getBodyRegion().getArgument(1);
    createEquationSides(nested, lhs, rhs);
  });

  auto equation = Equation::build(equationOp, variables);

  EXPECT_EQ(equation->getNumOfIterationVars(), 2);

  auto iterationRanges = equation->getIterationRanges();
  assert(iterationRanges.isSingleMultidimensionalRange());
  auto ranges = equation->getIterationRanges().minContainingRange();

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

  llvm::SmallVector<mlir::Type, 2> types;
  auto arrayType = ArrayType::get({ 3, 4 }, RealType::get(builder.getContext()));
  types.push_back(arrayType);
  types.push_back(arrayType);

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  auto equationOp = createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange indices) {
    mlir::Value lhs = model.getBodyRegion().getArgument(0);
    mlir::Value rhs = model.getBodyRegion().getArgument(1);
    createEquationSides(nested, lhs, rhs);
  });

  auto equation = Equation::build(equationOp, variables);
  auto accesses = equation->getAccesses();

  auto access1 = AccessMatcher(
      model.getBodyRegion().getArgument(0),
      AccessFunction({
          DimensionAccess::relative(0, 0),
          DimensionAccess::relative(1, 0)
      }),
      EquationPath(EquationPath::LEFT));

  auto access2 = AccessMatcher(
      model.getBodyRegion().getArgument(1),
      AccessFunction({
          DimensionAccess::relative(0, 0),
          DimensionAccess::relative(1, 0)
      }),
      EquationPath(EquationPath::RIGHT));

  EXPECT_THAT(accesses, testing::UnorderedElementsAre(access1, access2));
}
