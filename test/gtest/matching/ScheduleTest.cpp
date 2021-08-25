#include "gtest/gtest.h"
#include <mlir/Support/LogicalResult.h>
#include <marco/mlirlowerer/passes/matching/Matching.h>
#include <marco/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <marco/mlirlowerer/passes/matching/Schedule.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>

#include "../TestingUtils.h"

using namespace marco::codegen::model;

TEST(ScheduleTest, SimpleScheduling)
{
	std::string stringModel = "model Sched1 "
														"int[3] x; "
														"int[3] y; "
														"equation "
														"y[3] + 7 = x[2]; "
														"x[1] = 3 * y[2]; "
														"y[2] = x[3] + 2; "
														"x[1] + x[2] = 0; "
														"y[1] = -y[3]; "
														"x[3] = 7; "
														"end Sched1; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 2);
	EXPECT_EQ(model.getEquations().size(), 6);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	llvm::SmallVector<Equation, 3> equations;
	for (Equation eq : model.getEquations())
		equations.emplace_back(eq);

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 2);
	EXPECT_EQ(model.getEquations().size(), 6);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(schedule(model)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 2);
	EXPECT_EQ(model.getEquations().size(), 6);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	EXPECT_EQ(model.getEquations()[0], equations[5]);
	EXPECT_EQ(model.getEquations()[1], equations[2]);
	EXPECT_EQ(model.getEquations()[2], equations[1]);
	EXPECT_EQ(model.getEquations()[3], equations[3]);
	EXPECT_EQ(model.getEquations()[4], equations[0]);
	EXPECT_EQ(model.getEquations()[5], equations[4]);
}

TEST(ScheduleTest, EquationBeforeBltBlock)
{
	std::string stringModel = "model Sched2 "
														"int[3] x; "
														"equation "
														"x[1] + x[2] = 2; "
														"4 = x[3]; "
														"x[1] - x[2] = x[3]; "
														"end Sched2; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 3);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	Equation equation = model.getEquations()[1];

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 1);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	if (failed(schedule(model)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 1);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	EXPECT_EQ(model.getEquations()[0].amount(), 1);
	EXPECT_EQ(model.getEquations()[0], equation);
	EXPECT_EQ(model.getBltBlocks()[0].getEquations().size(), 2);
}

TEST(ScheduleTest, BltBlockBeforeEquation)
{
	std::string stringModel = "model Sched3 "
														"int[3] x; "
														"equation "
														"x[1] + x[2] = 2; "
														"x[2] = x[3]; "
														"x[1] - x[2] = 4; "
														"end Sched3; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 3);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	Equation equation = model.getEquations()[1];

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 1);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	if (failed(schedule(model)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 1);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	EXPECT_EQ(model.getEquations()[0].amount(), 1);
	EXPECT_EQ(model.getEquations()[0], equation);
	EXPECT_EQ(model.getBltBlocks()[0].getEquations().size(), 2);
}

TEST(ScheduleTest, MultipleBltBlocksAndEquations)
{
	std::string stringModel = "model Sched4 "
														"int[3] x; "
														"int[2] y; "
														"int[4] w; "
														"int z; "
														"equation "
														"for i in 3:3 loop "
														"w[i] - w[i+1] = y[2] - w[i-2]; "
														"end for; "
														"for j in 1:2 loop "
														"y[j] = w[j] + x[j]; "
														"end for; "
														"for j in 1:1 loop "
														"w[j] + w[j+1] = x[3]; "
														"end for; "
														"w[4] = 7 - w[3]; "
														"w[3] = z - y[1]; "
														"for i in 1:3 loop "
														"x[i] = 5; "
														"end for; "
														"w[1] - w[2] = 3; "
														"end Sched4; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_EQ(model.getVariables().size(), 4);
	EXPECT_EQ(model.getEquations().size(), 7);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 4);
	EXPECT_EQ(model.getEquations().size(), 7);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 4);
	EXPECT_EQ(model.getEquations().size(), 3);
	EXPECT_EQ(model.getBltBlocks().size(), 2);

	if (failed(schedule(model)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 4);
	EXPECT_EQ(model.getEquations().size(), 3);
	EXPECT_EQ(model.getBltBlocks().size(), 2);

	EXPECT_EQ(model.getEquations()[0].amount(), 3);
	EXPECT_EQ(model.getEquations()[1].amount(), 2);
	EXPECT_EQ(model.getEquations()[2].amount(), 1);
	EXPECT_EQ(model.getBltBlocks()[0].getEquations().size(), 2);
	EXPECT_EQ(model.getBltBlocks()[1].getEquations().size(), 2);
}

TEST(ScheduleTest, BltBlockAndVectorEquation)
{
	std::string stringModel = "model Sched5 "
														"int[5] z; "
														"int[10] x; "
														"int[2] y; "
														"equation "
														"y[1] + y[2] = x[3]; "
														"for i in 1:5 loop "
														"z[i] = x[i+5] - y[1]; "
														"end for; "
														"y[2] = x[7] + y[1]; "
														"for j in 1:10 loop "
														"5 = x[j]; "
														"end for; "
														"end Sched5; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 4);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 4);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 2);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	if (failed(schedule(model)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 2);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	EXPECT_EQ(model.getEquations()[0].amount(), 10);
	EXPECT_EQ(model.getEquations()[1].amount(), 5);
	EXPECT_EQ(model.getBltBlocks()[0].getEquations().size(), 2);
}

TEST(ScheduleTest, DerivativesOutsideScc)
{
	std::string stringModel = "model Sched6 "
														"Real[2] x; "
														"Real[2] y; "
														"equation "
														"x[1] + x[2] = der(y[2]); "
														"x[2] = der(y[1]); "
														"x[1] - x[2] = 4; "
														"der(y[2]) = 2; "
														"end Sched6; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 4);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	Equation equation1 = model.getEquations()[1];
	Equation equation2 = model.getEquations()[3];

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 2);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	if (failed(schedule(model)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 2);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	EXPECT_EQ(model.getEquations()[0].amount(), 1);
	EXPECT_EQ(model.getEquations()[0], equation2);
	EXPECT_EQ(model.getEquations()[1].amount(), 1);
	EXPECT_EQ(model.getEquations()[1], equation1);
	EXPECT_EQ(model.getBltBlocks()[0].getEquations().size(), 2);
}

TEST(ScheduleTest, DerivativesInsideScc)
{
	std::string stringModel = "model Sched7 "
														"Real x; "
														"Real y; "
														"Real z; "
														"equation "
														"der(y) + x = 2; "
														"z = 4; "
														"der(y) - x = z; "
														"end Sched7; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 4);
	EXPECT_EQ(model.getEquations().size(), 3);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	Equation equation = model.getEquations()[1];

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 4);
	EXPECT_EQ(model.getEquations().size(), 1);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	if (failed(schedule(model)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 4);
	EXPECT_EQ(model.getEquations().size(), 1);
	EXPECT_EQ(model.getBltBlocks().size(), 1);

	EXPECT_EQ(model.getEquations()[0].amount(), 1);
	EXPECT_EQ(model.getEquations()[0], equation);
	EXPECT_EQ(model.getBltBlocks()[0].getEquations().size(), 2);
}
