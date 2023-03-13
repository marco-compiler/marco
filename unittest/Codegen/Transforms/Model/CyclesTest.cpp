#include "gmock/gmock.h"
#include "llvm/ADT/STLExtras.h"
#include "marco/Codegen/Transforms/ModelSolving/CyclesSubstitutionSolver.h"
#include "marco/Modeling/Cycles.h"

#include "Utils.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

using namespace ::marco::codegen::test;

// Real x;
// Real y;
// Real z;
//
// x = y - 4
// y = 2 - z
// z = 3 * x
TEST(Cycles, solvableScalarCycleWithExplicitEquations)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  auto realType = RealType::get(builder.getContext());
  llvm::SmallVector<mlir::Type, 3> types(3, realType);

  auto model = createModel(builder, types);
  auto variables = discoverVariables(model);

  auto eq1 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");
    mlir::Value y = nested.create<VariableGetOp>(loc, types[1], "var1");

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        y,
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 4)));

    createEquationSides(nested, x, rhs);
  }), variables);

  auto eq1_matched = std::make_unique<MatchedEquation>(std::move(eq1), eq1->getIterationRanges(), EquationPath(EquationPath::LEFT));

  auto eq2 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value y = nested.create<VariableGetOp>(loc, types[1], "var1");
    mlir::Value z = nested.create<VariableGetOp>(loc, types[2], "var2");

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 2)),
        z);

    createEquationSides(nested, y, rhs);
  }), variables);

  auto eq2_matched = std::make_unique<MatchedEquation>(std::move(eq2), eq2->getIterationRanges(), EquationPath(EquationPath::LEFT));

  auto eq3 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value z = nested.create<VariableGetOp>(loc, types[2], "var2");
    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");

    mlir::Value rhs = nested.create<MulOp>(
        loc, realType,
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 3)),
        x);

    createEquationSides(nested, z, rhs);
  }), variables);

  auto eq3_matched = std::make_unique<MatchedEquation>(std::move(eq3), eq3->getIterationRanges(), EquationPath(EquationPath::LEFT));

  llvm::SmallVector<MatchedEquation*, 3> equations;
  equations.push_back(eq1_matched.get());
  equations.push_back(eq2_matched.get());
  equations.push_back(eq3_matched.get());

  CyclesFinder<Variable*, MatchedEquation*> cyclesFinder;
  cyclesFinder.addEquations(equations);

  auto cycles = cyclesFinder.getEquationsCycles();
  EXPECT_EQ(cycles.size(), 3);

  using Cycle = decltype(cyclesFinder)::Cycle;
  CyclesPermutation<Cycle> permutations(cycles);

  do {
    CyclesSubstitutionSolver<Cycle> solver(builder);

    for (const auto& cycle : cycles) {
      solver.solve(cycle);
    }

    EXPECT_EQ(solver.getUnsolvedEquations().size(), 0);
  } while (permutations.nextPermutation());
}

// Real[4] x;
// Real[4] y;
// Real[4] z;
//
// for i in 1:4 loop
//   x[i] = y[i] - 2;
// end for;
//
// for i in 1:4 loop
//   y[i] = z[i];
// end for;
//
// for i in 1:2 loop
//   z[i] = 3 * x[i] - 4;
// end for;
//
// for i in 3:4 loop
//   z[i] = 5 * x[i] - 6;
// end for;
TEST(Cycles, solvableArrayCycleWithBifurcation)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  auto realType = RealType::get(builder.getContext());
  auto arrayType = ArrayType::get({ 4 }, realType);
  llvm::SmallVector<mlir::Type, 3> types(3, arrayType);

  auto model = createModel(builder, types);
  auto variables = discoverVariables(model);

  std::vector<std::pair<long, long>> eq1_ranges;
  eq1_ranges.emplace_back(1, 4);

  auto eq1 = Equation::build(createEquation(builder, model, eq1_ranges, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");
    mlir::Value lhs = nested.create<LoadOp>(loc, x, inductions[0]);

    mlir::Value y = nested.create<VariableGetOp>(loc, types[1], "var1");

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<LoadOp>(loc, y, inductions[0]),
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 2)));

    createEquationSides(nested, lhs, rhs);
  }), variables);

  auto eq1_matched = std::make_unique<MatchedEquation>(std::move(eq1), eq1->getIterationRanges(), EquationPath(EquationPath::LEFT));

  std::vector<std::pair<long, long>> eq2_ranges;
  eq2_ranges.emplace_back(1, 4);

  auto eq2 = Equation::build(createEquation(builder, model, eq2_ranges, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value y = nested.create<VariableGetOp>(loc, types[1], "var1");
    mlir::Value z = nested.create<VariableGetOp>(loc, types[2], "var2");

    mlir::Value lhs = nested.create<LoadOp>(loc, y, inductions[0]);
    mlir::Value rhs = nested.create<LoadOp>(loc, z, inductions[0]);

    createEquationSides(nested, lhs, rhs);
  }), variables);

  auto eq2_matched = std::make_unique<MatchedEquation>(std::move(eq2), eq2->getIterationRanges(), EquationPath(EquationPath::LEFT));

  std::vector<std::pair<long, long>> eq3_ranges;
  eq3_ranges.emplace_back(1, 2);

  auto eq3 = Equation::build(createEquation(builder, model, eq3_ranges, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value z = nested.create<VariableGetOp>(loc, types[2], "var2");
    mlir::Value lhs = nested.create<LoadOp>(loc, z, inductions[0]);

    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<MulOp>(
            loc, realType,
            nested.create<ConstantOp>(loc, RealAttr::get(realType, 3)),
            x),
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 4)));

    createEquationSides(nested, lhs, rhs);
  }), variables);

  auto eq3_matched = std::make_unique<MatchedEquation>(std::move(eq3), eq3->getIterationRanges(), EquationPath(EquationPath::LEFT));

  std::vector<std::pair<long, long>> eq4_ranges;
  eq4_ranges.emplace_back(3, 4);

  auto eq4 = Equation::build(createEquation(builder, model, eq4_ranges, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value z = nested.create<VariableGetOp>(loc, types[2], "var2");
    mlir::Value lhs = nested.create<LoadOp>(loc, z, inductions[0]);

    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<MulOp>(
            loc, realType,
            nested.create<ConstantOp>(loc, RealAttr::get(realType, 5)),
            nested.create<LoadOp>(loc, x, inductions[0])),
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 6)));

    createEquationSides(nested, lhs, rhs);
  }), variables);

  auto eq4_matched = std::make_unique<MatchedEquation>(std::move(eq4), eq4->getIterationRanges(), EquationPath(EquationPath::LEFT));

  llvm::SmallVector<MatchedEquation*, 3> equations;
  equations.push_back(eq1_matched.get());
  equations.push_back(eq2_matched.get());
  equations.push_back(eq3_matched.get());
  equations.push_back(eq4_matched.get());

  CyclesFinder<Variable*, MatchedEquation*> cyclesFinder;
  cyclesFinder.addEquations(equations);

  auto cycles = cyclesFinder.getEquationsCycles();
  EXPECT_EQ(cycles.size(), 4);

  using Cycle = decltype(cyclesFinder)::Cycle;
  CyclesPermutation<Cycle> permutations(cycles);

  do {
    CyclesSubstitutionSolver<Cycle> solver(builder);

    for (const auto& cycle : cycles) {
      solver.solve(cycle);
    }

    EXPECT_EQ(solver.getUnsolvedEquations().size(), 0);
  } while (permutations.nextPermutation());
}

// Real x;
// Real y;
// Real z;
//
// x + sin(x) = y
// y = 2 - z
// z = 3 * x
TEST(Cycles, solvableScalarCycleWithImplicitEquation)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  auto realType = RealType::get(builder.getContext());
  llvm::SmallVector<mlir::Type, 3> types(3, realType);

  auto model = createModel(builder, types);
  auto variables = discoverVariables(model);

  auto eq1 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");
    mlir::Value y = nested.create<VariableGetOp>(loc, types[1], "var1");

    mlir::Value lhs = nested.create<AddOp>(
        loc, realType,
        x,
        nested.create<SinOp>(loc, realType, x));

    createEquationSides(nested, lhs, y);
  }), variables);

  auto eq1_matched = std::make_unique<MatchedEquation>(std::move(eq1), eq1->getIterationRanges(), EquationPath(EquationPath::LEFT, { 0 }));

  auto eq2 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange indutions) {
    mlir::Value y = nested.create<VariableGetOp>(loc, types[1], "var1");
    mlir::Value z = nested.create<VariableGetOp>(loc, types[2], "var2");

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 2)),
        z);

    createEquationSides(nested, y, rhs);
  }), variables);

  auto eq2_matched = std::make_unique<MatchedEquation>(std::move(eq2), eq2->getIterationRanges(), EquationPath(EquationPath::LEFT));

  auto eq3 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value z = nested.create<VariableGetOp>(loc, types[2], "var2");
    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");

    mlir::Value rhs = nested.create<MulOp>(
        loc, realType,
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 3)),
        x);

    createEquationSides(nested, z, rhs);
  }), variables);

  auto eq3_matched = std::make_unique<MatchedEquation>(std::move(eq3), eq3->getIterationRanges(), EquationPath(EquationPath::LEFT));

  llvm::SmallVector<MatchedEquation*, 3> equations;
  equations.push_back(eq1_matched.get());
  equations.push_back(eq2_matched.get());
  equations.push_back(eq3_matched.get());

  CyclesFinder<Variable*, MatchedEquation*> cyclesFinder;
  cyclesFinder.addEquations(equations);

  auto cycles = cyclesFinder.getEquationsCycles();
  EXPECT_EQ(cycles.size(), 3);

  using Cycle = decltype(cyclesFinder)::Cycle;
  CyclesPermutation<Cycle> permutations(cycles);

  do {
    CyclesSubstitutionSolver<Cycle> solver(builder);

    for (const auto& cycle : cycles) {
      solver.solve(cycle);
    }

    EXPECT_EQ(solver.getUnsolvedEquations().size(), 0);
  } while (permutations.nextPermutation());
}

// Real x;
// Real y;
// Real z;
//
// x = y + z + 12;
// y = 5 * x - 2;
// z = -2 * x - 4;
TEST(Cycles, solvableScalarCycleWithMultipleDependencies)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  auto realType = RealType::get(builder.getContext());
  llvm::SmallVector<mlir::Type, 3> types(3, realType);

  auto model = createModel(builder, types);
  auto variables = discoverVariables(model);

  auto eq1 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");
    mlir::Value y = nested.create<VariableGetOp>(loc, types[1], "var1");
    mlir::Value z = nested.create<VariableGetOp>(loc, types[2], "var2");

    mlir::Value rhs = nested.create<AddOp>(
        loc, realType,
        nested.create<AddOp>(loc, realType, y, z),
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 12)));

    createEquationSides(nested, x, rhs);
  }), variables);

  auto eq1_matched = std::make_unique<MatchedEquation>(std::move(eq1), eq1->getIterationRanges(), EquationPath(EquationPath::LEFT));

  auto eq2 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value y = nested.create<VariableGetOp>(loc, types[1], "var1");
    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<MulOp>(
            loc, realType,
            nested.create<ConstantOp>(loc, RealAttr::get(realType, 5)),
            x),
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 2)));

    createEquationSides(nested, y, rhs);
  }), variables);

  auto eq2_matched = std::make_unique<MatchedEquation>(std::move(eq2), eq2->getIterationRanges(), EquationPath(EquationPath::LEFT));

  auto eq3 = Equation::build(createEquation(builder, model, llvm::None, [&](mlir::OpBuilder& nested, mlir::ValueRange inductions) {
    mlir::Value z = nested.create<VariableGetOp>(loc, types[2], "var2");
    mlir::Value x = nested.create<VariableGetOp>(loc, types[0], "var0");

    mlir::Value rhs = nested.create<SubOp>(
        loc, realType,
        nested.create<MulOp>(
            loc, realType,
            nested.create<ConstantOp>(loc, RealAttr::get(realType, -2)),
            x),
        nested.create<ConstantOp>(loc, RealAttr::get(realType, 4)));

    createEquationSides(nested, z, rhs);
  }), variables);

  auto eq3_matched = std::make_unique<MatchedEquation>(std::move(eq3), eq3->getIterationRanges(), EquationPath(EquationPath::LEFT));

  llvm::SmallVector<MatchedEquation*, 3> equations;
  equations.push_back(eq1_matched.get());
  equations.push_back(eq2_matched.get());
  equations.push_back(eq3_matched.get());

  CyclesFinder<Variable*, MatchedEquation*> cyclesFinder;
  cyclesFinder.addEquations(equations);

  auto cycles = cyclesFinder.getEquationsCycles();
  EXPECT_EQ(cycles.size(), 3);

  using Cycle = decltype(cyclesFinder)::Cycle;
  CyclesPermutation<Cycle> permutations(cycles);

  do {
    CyclesSubstitutionSolver<Cycle> solver(builder);

    for (const auto& cycle : cycles) {
      solver.solve(cycle);
    }

    EXPECT_EQ(solver.getUnsolvedEquations().size(), 0);
  } while (permutations.nextPermutation());
}
