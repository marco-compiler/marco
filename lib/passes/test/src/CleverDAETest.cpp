#include "gtest/gtest.h"

#include "modelica/matching/SccCollapsing.hpp"
#include "modelica/matching/Schedule.hpp"
#include "modelica/model/ModBltBlock.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModParser.hpp"
#include "modelica/passes/CleverDAE.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

TEST(CleverDAETest, AddDifferentialEqToBltBlock)
{
	const string stringModel =
			"init "
			"der_x = FLOAT[10]call fill FLOAT[10](FLOAT[1]{0.000000e+00}) "
			"state x = FLOAT[10]call fill FLOAT[10](FLOAT[1]{0.000000e+00}) "
			"const tau = FLOAT[1]{5.000000e+00} "
			"template "
			"eq_0m0 FLOAT[1](* FLOAT[1](at FLOAT[10]der_x, INT[1]{0}), "
			"FLOAT[1]{5.000000e+00}) = FLOAT[1]{1.000000e+00} "
			"eq_1m1 FLOAT[1](* FLOAT[1](at FLOAT[10]der_x, INT[1](+ INT[1](ind "
			"INT[1]{0}), INT[1]{-1})), FLOAT[1]{5.000000e+00}) = FLOAT[1](* "
			"FLOAT[1](ind INT[1]{0}), FLOAT[1]{2.000000e+00}) "
			"update "
			"for [0,1]template eq_0m0 matched [0,0] "
			"for [2,11]template eq_1m1 matched [0,0] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	EXPECT_EQ(model->getVars().size(), 3);
	EXPECT_EQ(model->getEquations().size(), 2);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*model), 1000);
	if (!collapsedModel)
	{
		outs() << collapsedModel.takeError();
		FAIL();
	}

	EXPECT_EQ(collapsedModel->getVars().size(), 3);
	EXPECT_EQ(collapsedModel->getEquations().size(), 2);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 0);

	auto scheduled = schedule(move(*collapsedModel));
	auto assignModel = addBLTBlocks(scheduled);
	if (!assignModel)
	{
		outs() << assignModel.takeError();
		FAIL();
	}

	EXPECT_EQ(assignModel->getVars().size(), 3);
	EXPECT_EQ(assignModel->getUpdates().size(), 0);
	EXPECT_EQ(assignModel->getBltBlocks().size(), 2);
	for (auto& bltBlock : collapsedModel->getBltBlocks())
	{
		EXPECT_EQ(bltBlock.getVars().size(), 1);
		EXPECT_EQ(bltBlock.getEquations().size(), 1);
		EXPECT_EQ(bltBlock.getResidual().size(), 1);
		EXPECT_EQ(bltBlock.getJacobian().size(), 1);
		EXPECT_EQ(bltBlock.getJacobian().front().size(), 1);
	}
}

TEST(CleverDAETest, AddImplicitEqToBltBlock)
{
	EXPECT_TRUE(true);	// TODO
}
