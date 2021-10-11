#include "gtest/gtest.h"
#include <marco/mlirlowerer/passes/matching/Matching.h>
#include <marco/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/utils/Interval.hpp>
#include <mlir/Support/LogicalResult.h>

#include "../TestingUtils.h"

using namespace marco::codegen::model;

TEST(SCCCollapsingTest, EquationShouldBeNormalizable)
{
	std::string stringModel = "model Test "
														"int[2] x; "
														"int[2] y; "
														"equation "
														"for i in 1:2 loop "
														"x[i] = 3; "
														"end for; "
														"for i in 3:4 loop "
														"y[i-2] = x[i-2]; "
														"end for; "
														"end Test; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(
			model.getEquations()[1].getInductions(),
			marco::MultiDimInterval({ { 3, 5 } }));

	for (Equation& eq : model.getEquations())
		if (failed(eq.explicitate()))
			FAIL();

	for (Equation& eq : model.getEquations())
		eq.normalize();

	auto acc = AccessToVar::fromExp(model.getEquations()[0].lhs());
	EXPECT_TRUE(acc.getAccess().isIdentity());
}

TEST(SCCCollapsingTest, ThreeDepthNormalization)
{
	std::string stringModel = "model Test "
														"int[2, 3, 4] x; "
														"equation "
														"for i in 1:2 loop "
														"for j in 1:3 loop "
														"for k in 1:4 loop "
														"x[i, j, k] = 5; "
														"end for; "
														"end for; "
														"end for; "
														"end Test; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(
			model.getEquations()[0].getInductions(),
			marco::MultiDimInterval({ { 1, 3 }, { 1, 4 }, { 1, 5 } }));

	for (Equation& eq : model.getEquations())
		if (failed(eq.explicitate()))
			FAIL();

	for (Equation& eq : model.getEquations())
		eq.normalize();

	auto acc = AccessToVar::fromExp(model.getEquations()[0].lhs());
	EXPECT_TRUE(acc.getAccess().isIdentity());
}

TEST(SCCCollapsingTest, CyclesWithScalarsSolved)
{
	std::string stringModel = "model Loop2 "
														"int x; "
														"int y; "
														"int z; "
														"int w; "
														"int v; "
														"equation "
														"x + y = 9 - v; "
														"x - y = 3; "
														"z + w = 1 + v; "
														"z - w = -1; "
														"v = 4; "
														"end Loop2; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 5);
	EXPECT_EQ(model.getEquations().size(), 5);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 5);
	EXPECT_EQ(model.getEquations().size(), 5);
	EXPECT_EQ(model.getBltBlocks().size(), 0);
	for (const BltBlock& bltBlock : model.getBltBlocks())
		EXPECT_EQ(bltBlock.getEquations().size(), 2);
}

TEST(SCCCollapsingTest, CyclesWithVectorsInBltBlock)
{
	std::string stringModel = "model Loop6 "
														"int[5] x; "
														"equation "
														"x[1] + x[2] = 2; "
														"x[1] - x[2] = 4; "
														"x[3] + x[4] = 1; "
														"x[3] - x[4] = -1; "
														"x[5] = x[4] + x[1]; "
														"end Loop6; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 5);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 1);
	EXPECT_EQ(model.getBltBlocks().size(), 2);
	for (auto& bltBlock : model.getBltBlocks())
		EXPECT_EQ(bltBlock.getEquations().size(), 2);
}

TEST(SCCCollapsingTest, CyclesWithThreeEquationsSolved)
{
	std::string stringModel = "model Loop4 "
														"Real x; "
														"Real y; "
														"Real z; "
														"equation "
														"x + y = 1.0; "
														"x + z = 2.0; "
														"y + z = 3.0; "
														"end Loop4; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 3);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 3);
	EXPECT_EQ(model.getBltBlocks().size(), 0);
}

TEST(SCCCollapsingTest, CyclesWithThreeDenseEquations)
{
	std::string stringModel = "model Loop16 "
														"Real[2] x; "
														"Real[2] y; "
														"Real[2] z; "
														"equation "
														"for i in 1:2 loop "
														"x[i] + y[i] - z[i] = 1.0; "
														"x[i] - y[i] + z[i] = 2.0; "
														"-x[i] + y[i] + z[i] = 3.0; "
														"end for; "
														"end Loop16; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 3);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 3);
	EXPECT_EQ(model.getEquations().size(), 0);
	EXPECT_EQ(model.getBltBlocks().size(), 1);
}

TEST(SCCCollapsingTest, CyclesWithForEquationsInBltBlock)
{
	std::string stringModel = "model Loop12 "
														"Real[4] x; "
														"equation "
														"for i in 1:1 loop "
														"x[i] + x[i+1] = x[4]; "
														"end for; "
														"for i in 1:1 loop "
														"x[i] - x[i+1] = 3.0; "
														"end for; "
														"x[2] = x[3]; "
														"2.0 = x[4]; "
														"end Loop12; ";

	mlir::MLIRContext context;
	Model model;
	makeModel(context, stringModel, model);

	if (failed(match(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 4);
	EXPECT_EQ(model.getBltBlocks().size(), 0);

	if (failed(solveSCCs(model, 1000)))
		FAIL();

	EXPECT_EQ(model.getVariables().size(), 1);
	EXPECT_EQ(model.getEquations().size(), 2);
	EXPECT_EQ(model.getBltBlocks().size(), 1);
	EXPECT_EQ(model.getBltBlocks()[0].getEquations().size(), 2);
}
