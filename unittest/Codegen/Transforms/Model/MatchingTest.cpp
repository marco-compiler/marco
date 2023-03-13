#include "gtest/gtest.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"

#include "Utils.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

using namespace ::marco::codegen::test;
using ::testing::Return;

TEST(MatchedEquation, inductionVariables)
{
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  // Create the model operation and map the variables
  llvm::SmallVector<mlir::Type, 2> types;
  types.push_back(RealType::get(builder.getContext()));
  types.push_back(RealType::get(builder.getContext()));

  auto model = createModel(builder, types);
  auto variables = discoverVariables(model);

  std::vector<std::pair<long, long>> equationRanges;
  equationRanges.emplace_back(2, 7);
  equationRanges.emplace_back(13, 23);

  auto equationOp = createEquation(builder, model, equationRanges, [&](mlir::OpBuilder& nested, mlir::ValueRange indices) {
    mlir::Value x = nested.create<VariableGetOp>(nested.getUnknownLoc(), types[0], "var0");
    mlir::Value y = nested.create<VariableGetOp>(nested.getUnknownLoc(), types[1], "var1");
    createEquationSides(nested, x, y);
  });

  auto equation = Equation::build(equationOp, variables);

  MultidimensionalRange matchedIndices({
    Range(3, 5),
    Range(14, 21)
  });

  EquationPath path(EquationPath::LEFT);
  MatchedEquation matchedEquation(std::move(equation), IndexSet(matchedIndices), path);

  IndexSet actual(matchedEquation.getIterationRanges());
  IndexSet expected(matchedIndices);

  EXPECT_EQ(actual, expected);
}
