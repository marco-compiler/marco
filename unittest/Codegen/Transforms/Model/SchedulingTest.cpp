#include "gtest/gtest.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"

#include "Utils.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

using namespace ::marco::codegen::test;
using ::testing::Return;

TEST(ScheduledEquation, inductionVariables)
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
    mlir::Value loadX = builder.create<LoadOp>(loc, model.getBodyRegion().getArgument(0), mlir::ValueRange({ indices[0], indices[1] }));
    mlir::Value loadY = builder.create<LoadOp>(loc, model.getBodyRegion().getArgument(1), mlir::ValueRange({ indices[0], indices[1] }));
    createEquationSides(builder, loadX, loadY);
  });

  auto equation = Equation::build(equationOp, variables);

  MultidimensionalRange matchedIndices({
    Range(3, 5),
    Range(14, 21)
  });

  EquationPath path(EquationPath::LEFT);
  auto matchedEquation = std::make_unique<MatchedEquation>(std::move(equation), IndexSet(matchedIndices), path);

  MultidimensionalRange scheduledIndices({
    Range(3, 4),
    Range(15, 17)
  });

  ScheduledEquation scheduledEquation(
      std::move(matchedEquation), IndexSet(scheduledIndices), ::marco::modeling::scheduling::Direction::Forward);

  IndexSet actual(scheduledEquation.getIterationRanges());
  IndexSet expected(scheduledIndices);

  EXPECT_EQ(actual, expected);
}
