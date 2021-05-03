#include "gtest/gtest.h"

#include "modelica/matching/SccCollapsing.hpp"
#include "modelica/model/ModBltBlock.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModParser.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

TEST(ModBltBlockTest, CyclesWithScalarsInBltBlock)
{
	const string stringModel =
			"init "
			"v = INT[1]{0} "
			"w = INT[1]{0} "
			"x = INT[1]{0} "
			"y = INT[1]{0} "
			"z = INT[1]{0} "
			"template "
			"eq_0m0 INT[1](+ INT[1]x, INT[1]y) = INT[1](+ INT[1](* INT[1]v, "
			"INT[1]{-1}), INT[1]{9}) "
			"eq_1m1 INT[1](+ INT[1]x, INT[1](* INT[1]y, INT[1]{-1})) = INT[1]{3} "
			"eq_2m2 INT[1](+ INT[1]z, INT[1]w) = INT[1](+ INT[1]v, INT[1]{1}) "
			"eq_3m3 INT[1](+ INT[1]z, INT[1](* INT[1]w, INT[1]{-1})) = INT[1]{-1} "
			"eq_4m4 INT[1]v = INT[1]{4} "
			"update "
			"for [0,1]template eq_0m0 matched [0,1] "
			"for [0,1]template eq_1m1 matched [0,0] "
			"for [0,1]template eq_2m2 matched [0,1] "
			"for [0,1]template eq_3m3 matched [0,0] "
			"for [0,1]template eq_4m4 matched [0] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	EXPECT_EQ(model->getVars().size(), 5);
	EXPECT_EQ(model->getEquations().size(), 5);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*model), 1000);
	if (!collapsedModel)
	{
		outs() << collapsedModel.takeError();
		FAIL();
	}

	EXPECT_EQ(collapsedModel->getVars().size(), 5);
	EXPECT_EQ(collapsedModel->getEquations().size(), 1);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 2);
	for (auto& bltBlock : collapsedModel->getBltBlocks())
	{
		EXPECT_EQ(bltBlock.getVars().size(), 2);
		EXPECT_EQ(bltBlock.getEquations().size(), 2);
		EXPECT_EQ(bltBlock.getResidual().size(), 2);
		EXPECT_EQ(bltBlock.getJacobian().size(), 2);
		EXPECT_EQ(bltBlock.getJacobian().front().size(), 2);
	}
}

TEST(ModBltBlockTest, CyclesWithVectorsInBltBlock)
{
	const string stringModel =
			"init "
			"x = INT[5]call fill INT[5](INT[1]{0}) "
			"template "
			"eq_0m0 INT[1](+ INT[1](at INT[5]x, INT[1]{0}), INT[1](at INT[5]x, "
			"INT[1]{1})) = INT[1]{2} "
			"eq_1m1 INT[1](+ INT[1](at INT[5]x, INT[1]{0}), INT[1](* INT[1](at "
			"INT[5]x, INT[1]{1}), INT[1]{-1})) = INT[1]{4} "
			"eq_2m2 INT[1](+ INT[1](at INT[5]x, INT[1]{2}), INT[1](at INT[5]x, "
			"INT[1]{3})) = INT[1]{1} "
			"eq_3m3 INT[1](+ INT[1](at INT[5]x, INT[1]{2}), INT[1](* INT[1](at "
			"INT[5]x, INT[1]{3}), INT[1]{-1})) = INT[1]{-1} "
			"eq_4m4 INT[1](at INT[5]x, INT[1]{4}) = INT[1](+ INT[1](at INT[5]x, "
			"INT[1]{3}), INT[1](at INT[5]x, INT[1]{0})) "
			"update "
			"for [0,1]template eq_0m0 matched [0,1] "
			"for [0,1]template eq_1m1 matched [0,0] "
			"for [0,1]template eq_2m2 matched [0,1] "
			"for [0,1]template eq_3m3 matched [0,0] "
			"for [0,1]template eq_4m4 matched [0] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	EXPECT_EQ(model->getVars().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 5);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*model), 1000);
	if (!collapsedModel)
	{
		outs() << collapsedModel.takeError();
		FAIL();
	}

	EXPECT_EQ(collapsedModel->getVars().size(), 1);
	EXPECT_EQ(collapsedModel->getEquations().size(), 1);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 2);
	for (auto& bltBlock : collapsedModel->getBltBlocks())
	{
		EXPECT_EQ(bltBlock.getVars().size(), 1);
		EXPECT_EQ(bltBlock.getEquations().size(), 2);
		EXPECT_EQ(bltBlock.getResidual().size(), 2);
		EXPECT_EQ(bltBlock.getJacobian().size(), 2);
		EXPECT_EQ(bltBlock.getJacobian().front().size(), 1);
	}
}

TEST(ModBltBlockTest, CycleMoreThanTwoEquations)
{
	const string stringModel =
			"init "
			"x = FLOAT[3]call fill FLOAT[3](INT[1]{0}) "
			"template "
			"eq_0m0 FLOAT[1](+ FLOAT[1](at FLOAT[3]x, INT[1]{0}), FLOAT[1](at "
			"FLOAT[3]x, INT[1]{1})) = FLOAT[1]{2.000000e+00} "
			"eq_1m1 FLOAT[1](+ FLOAT[1](at FLOAT[3]x, INT[1]{0}), FLOAT[1](* "
			"FLOAT[1](at FLOAT[3]x, INT[1]{2}), FLOAT[1]{-1.000000e+00})) = "
			"FLOAT[1]{3.000000e+00} "
			"eq_2m2 FLOAT[1](+ FLOAT[1](at FLOAT[3]x, INT[1]{1}), FLOAT[1](at "
			"FLOAT[3]x, INT[1]{2})) = FLOAT[1]{1.000000e+00} "
			"update "
			"for [0,1]template eq_0m0 matched [0,1] "
			"for [0,1]template eq_1m1 matched [0,0] "
			"for [0,1]template eq_2m2 matched [0,1] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	EXPECT_EQ(model->getVars().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 3);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*model), 1000);
	if (!collapsedModel)
	{
		outs() << collapsedModel.takeError();
		FAIL();
	}

	EXPECT_EQ(collapsedModel->getVars().size(), 1);
	EXPECT_EQ(collapsedModel->getEquations().size(), 0);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 1);
	EXPECT_EQ(collapsedModel->getBltBlock(0).getVars().size(), 1);
	EXPECT_EQ(collapsedModel->getBltBlock(0).getEquations().size(), 3);
	EXPECT_EQ(collapsedModel->getBltBlock(0).getResidual().size(), 3);
	EXPECT_EQ(collapsedModel->getBltBlock(0).getJacobian().size(), 3);
	EXPECT_EQ(collapsedModel->getBltBlock(0).getJacobian().front().size(), 1);
}
