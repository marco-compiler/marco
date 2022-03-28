#include "gtest/gtest.h"
#include "marco/Codegen/Transforms/Model/Scheduling.h"

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
  auto variables = mapVariables(model);

  // Create the loops
  auto loop1 = builder.create<ForEquationOp>(loc, 2, 7);
  builder.setInsertionPointToStart(loop1.bodyBlock());

  auto loop2 = builder.create<ForEquationOp>(loc, 13, 23);
  builder.setInsertionPointToStart(loop2.bodyBlock());

  // Create the equation body
  auto equationOp = builder.create<EquationOp>(loc);
  builder.setInsertionPointToStart(equationOp.bodyBlock());

  mlir::Value loadX = builder.create<LoadOp>(loc, model.bodyRegion().getArgument(0), mlir::ValueRange({ loop1.induction(), loop2.induction() }));
  mlir::Value loadY = builder.create<LoadOp>(loc, model.bodyRegion().getArgument(1), mlir::ValueRange({ loop1.induction(), loop2.induction() }));
  builder.create<EquationSidesOp>(builder.getUnknownLoc(), loadX, loadY);

  auto equation = Equation::build(equationOp, variables);

  MultidimensionalRange matchedIndices({
    Range(3, 5),
    Range(14, 21)
  });

  EquationPath path(EquationPath::LEFT);
  auto matchedEquation = std::make_unique<MatchedEquation>(std::move(equation), matchedIndices, path);

  MultidimensionalRange scheduledIndices({
    Range(3, 4),
    Range(15, 17)
  });

  ScheduledEquation scheduledEquation(
      std::move(matchedEquation), scheduledIndices, ::marco::modeling::scheduling::Direction::Forward);

  IndexSet actual(scheduledEquation.getIterationRanges());
  IndexSet expected(scheduledIndices);

  EXPECT_EQ(actual, expected);
}
