#include "gmock/gmock.h"
#include "llvm/ADT/STLExtras.h"
#include "marco/Codegen/Transforms/Model/CyclesLinearSolver.h"
#include "marco/Codegen/Transforms/Model/Matching.h"
#include "marco/Modeling/Cycles.h"

#include "Utils.h"

using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

using namespace ::marco::codegen::test;

// Real x;
// Real y;
// Real z;
//
// x = y - 4
// y = 2 - z
// z = 3 * x
TEST(Cycles, solvableCycleWithExplicitEquations)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  auto realType = RealType::get(builder.getContext());
  llvm::SmallVector<mlir::Type, 3> types(3, realType);

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  auto eq1 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested) {
    mlir::Value lhs = nested.create<LoadOp>(loc, model.body().getArgument(0));

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<LoadOp>(loc, model.body().getArgument(1)),
        nested.create<ConstantOp>(loc, RealAttribute::get(realType, 4)));

    nested.create<EquationSidesOp>(loc, lhs, rhs);
  }), variables);

  auto eq1_matched = std::make_unique<MatchedEquation>(std::move(eq1), eq1->getIterationRanges(), EquationPath(EquationPath::LEFT));

  auto eq2 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested) {
    mlir::Value lhs = nested.create<LoadOp>(loc, model.body().getArgument(1));

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<ConstantOp>(loc, RealAttribute::get(realType, 2)),
        nested.create<LoadOp>(loc, model.body().getArgument(2)));

    nested.create<EquationSidesOp>(loc, lhs, rhs);
  }), variables);

  auto eq2_matched = std::make_unique<MatchedEquation>(std::move(eq2), eq2->getIterationRanges(), EquationPath(EquationPath::LEFT));

  auto eq3 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested) {
    mlir::Value lhs = nested.create<LoadOp>(loc, model.body().getArgument(2));

    mlir::Value rhs = nested.create<MulOp>(
        loc, realType,
        nested.create<ConstantOp>(loc, RealAttribute::get(realType, 3)),
        nested.create<LoadOp>(loc, model.body().getArgument(0)));

    nested.create<EquationSidesOp>(loc, lhs, rhs);
  }), variables);

  auto eq3_matched = std::make_unique<MatchedEquation>(std::move(eq3), eq3->getIterationRanges(), EquationPath(EquationPath::LEFT));

  llvm::SmallVector<MatchedEquation*, 3> equations;
  equations.push_back(eq1_matched.get());
  equations.push_back(eq2_matched.get());
  equations.push_back(eq3_matched.get());

  CyclesFinder<Variable*, MatchedEquation*> cyclesFinder(equations);
  auto cycles = cyclesFinder.getEquationsCycles();
  EXPECT_EQ(cycles.size(), 3);

  CyclesLinearSolver<decltype(cyclesFinder)::Cycle> solver(builder);

  for (const auto& cycle : cycles) {
    solver.solve(cycle);
  }

  EXPECT_EQ(solver.getSolvedEquations().size(), 3);
  EXPECT_EQ(solver.getUnsolvedEquations().size(), 0);
}

// Real x;
// Real y;
// Real z;
//
// x + sin(x) = y
// y = 2 - z
// z = 3 * x
TEST(Cycles, solvableCycleWithImplicitEquation)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  auto realType = RealType::get(builder.getContext());
  llvm::SmallVector<mlir::Type, 3> types(3, realType);

  auto model = createModel(builder, types);
  auto variables = mapVariables(model);

  auto eq1 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested) {
    mlir::Value lhs = nested.create<AddOp>(
        loc, realType,
        nested.create<LoadOp>(loc, model.body().getArgument(0)),
        nested.create<SinOp>(loc, realType, nested.create<LoadOp>(loc, model.body().getArgument(0))));

    mlir::Value rhs = nested.create<LoadOp>(loc, model.body().getArgument(1));
    nested.create<EquationSidesOp>(loc, lhs, rhs);
  }), variables);

  auto eq1_matched = std::make_unique<MatchedEquation>(std::move(eq1), eq1->getIterationRanges(), EquationPath(EquationPath::LEFT, { 0 }));

  auto eq2 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested) {
    mlir::Value lhs = nested.create<LoadOp>(loc, model.body().getArgument(1));

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<ConstantOp>(loc, RealAttribute::get(realType, 2)),
        nested.create<LoadOp>(loc, model.body().getArgument(2)));

    nested.create<EquationSidesOp>(loc, lhs, rhs);
  }), variables);

  auto eq2_matched = std::make_unique<MatchedEquation>(std::move(eq2), eq2->getIterationRanges(), EquationPath(EquationPath::LEFT));

  auto eq3 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested) {
    mlir::Value lhs = nested.create<LoadOp>(loc, model.body().getArgument(2));

    mlir::Value rhs = nested.create<MulOp>(
        loc, realType,
        nested.create<ConstantOp>(loc, RealAttribute::get(realType, 3)),
        nested.create<LoadOp>(loc, model.body().getArgument(0)));

    nested.create<EquationSidesOp>(loc, lhs, rhs);
  }), variables);

  auto eq3_matched = std::make_unique<MatchedEquation>(std::move(eq3), eq3->getIterationRanges(), EquationPath(EquationPath::LEFT));

  llvm::SmallVector<MatchedEquation*, 3> equations;
  equations.push_back(eq1_matched.get());
  equations.push_back(eq2_matched.get());
  equations.push_back(eq3_matched.get());

  CyclesFinder<Variable*, MatchedEquation*> cyclesFinder(equations);
  auto cycles = cyclesFinder.getEquationsCycles();
  EXPECT_EQ(cycles.size(), 3);

  CyclesLinearSolver<decltype(cyclesFinder)::Cycle> solver(builder);

  for (const auto& cycle : cycles) {
    solver.solve(cycle);
  }

  EXPECT_EQ(solver.getUnsolvedEquations().size(), 0);
}
