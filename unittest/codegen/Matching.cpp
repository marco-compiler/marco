#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/codegen/passes/model/Model.h"

using namespace marco::codegen;
using namespace marco::codegen::modelica;
/*
static ModelOp createModel(mlir::OpBuilder& builder, mlir::TypeRange varTypes)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  llvm::SmallVector<mlir::Attribute, 3> names;

  for (size_t i = 0; i < varTypes.size(); ++i)
    names.push_back(builder.getStringAttr("var" + std::to_string(i)));

  llvm::SmallVector<mlir::Type, 3> varArrayTypes;

  for (const auto& type : varTypes)
  {
    if (auto arrayType = type.dyn_cast<ArrayType>())
      varArrayTypes.push_back(arrayType.toAllocationScope(BufferAllocationScope::unknown));
    else
      varArrayTypes.push_back(ArrayType::get(builder.getContext(), BufferAllocationScope::unknown, type));
  }

  auto model = builder.create<ModelOp>(
          builder.getUnknownLoc(),
          builder.getArrayAttr(names),
          RealAttribute::get(builder.getContext(), 0),
          RealAttribute::get(builder.getContext(), 10),
          RealAttribute::get(builder.getContext(), 0.1),
          varArrayTypes);

  builder.setInsertionPointToStart(&model.init().front());
  llvm::SmallVector<mlir::Value, 3> members;

  for (const auto& [name, type] : llvm::zip(names, varArrayTypes))
  {
    auto arrayType = type.cast<ArrayType>().toAllocationScope(BufferAllocationScope::heap);
    auto memberType = MemberType::get(arrayType);
    auto member = builder.create<MemberCreateOp>(builder.getUnknownLoc(), name.cast<mlir::StringAttr>().getValue(), memberType, llvm::None);
    members.push_back(member.getResult());
  }

  builder.create<YieldOp>(builder.getUnknownLoc(), members);
  return model;
}

void mapVariables(ModelOp model, llvm::SmallVectorImpl<Variable>& variables)
{
  for (const auto& variable : model.body().getArguments())
    variables.emplace_back(variable);
}

TEST(Codegen, matching_scalarVariable)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);

  auto type = RealType::get(builder.getContext());
  auto model = createModel(builder, type);

  llvm::SmallVector<Variable, 3> variables;
  mapVariables(model, variables);

  EXPECT_EQ(variables.size(), 1);
  EXPECT_EQ(variables[0].getRank(), 1);
  EXPECT_EQ(variables[0].getDimensionSize(0), 1);
}

TEST(Codegen, matching_arrayVariable)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);

  llvm::SmallVector<long, 3> shape;
  shape.push_back(3);
  shape.push_back(4);
  shape.push_back(5);

  auto baseType = RealType::get(builder.getContext());
  auto arrayType = ArrayType::get(builder.getContext(), BufferAllocationScope::heap, baseType, shape);
  auto model = createModel(builder, arrayType);

  llvm::SmallVector<Variable, 3> variables;
  mapVariables(model, variables);

  EXPECT_EQ(variables.size(), 1);
  EXPECT_EQ(variables[0].getRank(), 3);
  EXPECT_EQ(variables[0].getDimensionSize(0), 3);
  EXPECT_EQ(variables[0].getDimensionSize(1), 4);
  EXPECT_EQ(variables[0].getDimensionSize(2), 5);
}

/// Real x;
/// Real y;
///
/// x = y
TEST(Codegen, matching_scalarEquationInductionVars)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);

  llvm::SmallVector<mlir::Type, 2> types;
  types.push_back(RealType::get(builder.getContext()));
  types.push_back(RealType::get(builder.getContext()));

  auto model = createModel(builder, types);

  llvm::SmallVector<Variable, 3> variables;
  mapVariables(model, variables);

  builder.setInsertionPointToStart(&model.body().front());
  auto equationOp = builder.create<EquationOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(equationOp.body());

  mlir::Value loadX = builder.create<LoadOp>(builder.getUnknownLoc(), model.body().getArgument(0));
  mlir::Value loadY = builder.create<LoadOp>(builder.getUnknownLoc(), model.body().getArgument(0));
  builder.create<EquationSidesOp>(builder.getUnknownLoc(), loadX, loadY);

  Equation equation(equationOp, variables);
  EXPECT_EQ(equation.getNumOfIterationVars(), 1);
  EXPECT_EQ(equation.getRangeStart(0), 0);
  EXPECT_EQ(equation.getRangeEnd(0), 1);
}

/// Real x;
/// Real y;
///
/// x = y
TEST(Codegen, matching_scalarEquationAccesses)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);

  llvm::SmallVector<mlir::Type, 2> types;
  types.push_back(RealType::get(builder.getContext()));
  types.push_back(RealType::get(builder.getContext()));

  auto model = createModel(builder, types);

  llvm::SmallVector<Variable, 3> variables;
  mapVariables(model, variables);

  builder.setInsertionPointToStart(&model.body().front());
  auto equationOp = builder.create<EquationOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(equationOp.body());

  mlir::Value loadX = builder.create<LoadOp>(builder.getUnknownLoc(), model.body().getArgument(0));
  mlir::Value loadY = builder.create<LoadOp>(builder.getUnknownLoc(), model.body().getArgument(0));
  builder.create<EquationSidesOp>(builder.getUnknownLoc(), loadX, loadY);

  Equation equation(equationOp, variables);
  llvm::SmallVector<Equation::Access, 3> accesses;
  equation.getVariableAccesses(accesses);

  EXPECT_EQ(accesses.size(), 2);

  auto xAccesses = accesses[0].getAccessFunction();
  EXPECT_EQ(xAccesses.size(), 1);
  EXPECT_TRUE(xAccesses[0].isConstantAccess());
  EXPECT_EQ(xAccesses[0].getPosition(), 0);

  auto yAccesses = accesses[1].getAccessFunction();
  EXPECT_EQ(yAccesses.size(), 1);
  EXPECT_TRUE(yAccesses[0].isConstantAccess());
  EXPECT_EQ(yAccesses[0].getPosition(), 0);
}

/// Real[4,5] x;
/// Real[6] y;
///
/// for i in 2:5 loop
///   for j in 2:4 loop
///     x[i - 1, j + 2] = y[i];
///   end for;
/// end for;
TEST(Codegen, matching_equationWithExplicitLoops)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type, 2> types;
  types.push_back(ArrayType::get(builder.getContext(), BufferAllocationScope::heap, RealType::get(builder.getContext()), { 4, 5 }));
  types.push_back(ArrayType::get(builder.getContext(), BufferAllocationScope::heap, RealType::get(builder.getContext()), { 6 }));

  auto model = createModel(builder, types);

  llvm::SmallVector<Variable, 3> variables;
  mapVariables(model, variables);

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

  Equation equation(equationOp, variables);
  llvm::SmallVector<Equation::Access, 3> accesses;
  equation.getVariableAccesses(accesses);

  EXPECT_EQ(accesses.size(), 2);

  auto xAccesses = accesses[0].getAccessFunction();
  EXPECT_EQ(xAccesses.size(), 2);

  EXPECT_FALSE(xAccesses[0].isConstantAccess());
  EXPECT_EQ(xAccesses[0].getInductionVariableIndex(), 0);
  EXPECT_EQ(xAccesses[0].getOffset(), -1);

  EXPECT_FALSE(xAccesses[1].isConstantAccess());
  EXPECT_EQ(xAccesses[1].getInductionVariableIndex(), 1);
  EXPECT_EQ(xAccesses[1].getOffset(), 2);

  auto yAccesses = accesses[1].getAccessFunction();
  EXPECT_EQ(yAccesses.size(), 1);
  EXPECT_FALSE(yAccesses[0].isConstantAccess());
  EXPECT_EQ(yAccesses[0].getInductionVariableIndex(), 0);
  EXPECT_EQ(yAccesses[0].getOffset(), 0);
}

/// Real[3] x;
/// Real[3] y;
///
/// x = y;
TEST(Codegen, matching_equationWithImplicitLoops)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type, 2> types;
  auto arrayType = ArrayType::get(builder.getContext(), BufferAllocationScope::heap, RealType::get(builder.getContext()), { 3 });
  types.push_back(arrayType);
  types.push_back(arrayType);

  auto model = createModel(builder, types);

  llvm::SmallVector<Variable, 3> variables;
  mapVariables(model, variables);

  builder.setInsertionPointToStart(&model.body().front());
  auto equationOp = builder.create<EquationOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(equationOp.body());

  builder.create<EquationSidesOp>(loc, model.body().getArgument(0), model.body().getArgument(1));

  Equation equation(equationOp, variables);
  EXPECT_EQ(equation.getNumOfIterationVars(), 1);
  EXPECT_EQ(equation.getRangeStart(0), 0);
  EXPECT_EQ(equation.getRangeEnd(0), 3);

  llvm::SmallVector<Equation::Access, 3> accesses;
  equation.getVariableAccesses(accesses);

  EXPECT_EQ(accesses.size(), 2);

  auto xAccesses = accesses[0].getAccessFunction();
  EXPECT_EQ(xAccesses.size(), 1);
  EXPECT_FALSE(xAccesses[0].isConstantAccess());
  EXPECT_EQ(xAccesses[0].getInductionVariableIndex(), 0);
  EXPECT_EQ(xAccesses[0].getOffset(), 0);

  auto yAccesses = accesses[1].getAccessFunction();
  EXPECT_EQ(yAccesses.size(), 1);
  EXPECT_FALSE(yAccesses[0].isConstantAccess());
  EXPECT_EQ(yAccesses[0].getInductionVariableIndex(), 0);
  EXPECT_EQ(yAccesses[0].getOffset(), 0);
}
*/