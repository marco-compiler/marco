#include "gtest/gtest.h"

#include "marco/matching/SccCollapsing.hpp"
#include "marco/matching/Schedule.hpp"
#include "marco/model/ModBltBlock.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModParser.hpp"
#include "marco/passes/CleverDAE.hpp"

using namespace std;
using namespace llvm;
using namespace marco;

TEST(CleverDAETest, AddDifferentialEqToBltBlock)
{
	/* Tested Model:
		model SimpleDer
			final parameter Real tau = 5.0;
			Real[10] x(start = 0.0);
		equation
			tau*der(x[1]) = 1.0;
			for i in 2:10 loop
				tau*der(x[i]) = 2.0*i;
			end for;
		end SimpleDer;
	*/

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

	ScheduledModel scheduledModel = ScheduledModel(model->getVars());
	for (ModEquation eq : model->getEquations())
		scheduledModel.addUpdate(eq);
	for (ModBltBlock bltBlock : model->getBltBlocks())
		scheduledModel.addUpdate(bltBlock);

	auto assignModel = addBltBlocks(scheduledModel);
	if (!assignModel)
	{
		outs() << assignModel.takeError();
		FAIL();
	}

	EXPECT_EQ(assignModel->getVars().size(), 3);
	EXPECT_EQ(assignModel->getUpdates().size(), 2);
	for (auto& update : assignModel->getUpdates())
	{
		EXPECT_TRUE(holds_alternative<ModBltBlock>(update));
		ModBltBlock bltBlock = get<ModBltBlock>(update);

		EXPECT_EQ(bltBlock.getEquations().size(), 1);
		EXPECT_EQ(bltBlock.getResidual().size(), 1);
		EXPECT_EQ(bltBlock.getJacobian().size(), 1);
		EXPECT_EQ(bltBlock.getJacobian().front().size(), 1);
	}
}

TEST(CleverDAETest, AddImplicitEqToBltBlock1)
{
	/* Tested Model:
		model Implicit1
			Real x;
		equation
			x*x*x - x + 2 = 3;
		end Implicit1;
	*/

	const string stringModel =
			"init "
			"x = FLOAT[1]{0.000000e+00} "
			"template "
			"eq_0m0 FLOAT[1](+ FLOAT[1](* FLOAT[1]x, FLOAT[1]{-1.000000e+00}), "
			"FLOAT[1](+ FLOAT[1](* FLOAT[1](* FLOAT[1]x, FLOAT[1]x), FLOAT[1]x), "
			"FLOAT[1]{2.000000e+00})) = INT[1]{3} "
			"update "
			"for [0,1]template eq_0m0 matched [0,1,0,1] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	EXPECT_EQ(model->getVars().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 1);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	ScheduledModel scheduledModel = ScheduledModel(model->getVars());
	for (ModEquation eq : model->getEquations())
		scheduledModel.addUpdate(eq);
	for (ModBltBlock bltBlock : model->getBltBlocks())
		scheduledModel.addUpdate(bltBlock);

	auto assignModel = addBltBlocks(scheduledModel);
	if (!assignModel)
	{
		outs() << assignModel.takeError();
		FAIL();
	}

	EXPECT_EQ(assignModel->getVars().size(), 1);
	EXPECT_EQ(assignModel->getUpdates().size(), 1);
	EXPECT_TRUE(holds_alternative<ModBltBlock>(assignModel->getUpdates()[0]));

	ModBltBlock bltBlock = get<ModBltBlock>(assignModel->getUpdates()[0]);
	EXPECT_EQ(bltBlock.getEquations().size(), 1);
	EXPECT_EQ(bltBlock.getResidual().size(), 1);
	EXPECT_EQ(bltBlock.getJacobian().size(), 1);
	EXPECT_EQ(bltBlock.getJacobian().front().size(), 1);
}

TEST(CleverDAETest, AddImplicitEqToBltBlock2)
{
	/* Tested Model:
		model Implicit2
			Real x;
		equation
			x - 1/x + 2 = 3;
		end Implicit2;
	*/

	const string stringModel =
			"init "
			"x = FLOAT[1]{0.000000e+00} "
			"template "
			"eq_0m0 FLOAT[1](+ FLOAT[1](* FLOAT[1](/ FLOAT[1]{1.000000e+00}, "
			"FLOAT[1]x), FLOAT[1]{-1.000000e+00}), FLOAT[1](+ FLOAT[1]x, "
			"FLOAT[1]{2.000000e+00})) = INT[1]{3} "
			"update "
			"for [0,1]template eq_0m0 matched [0,1,0] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	EXPECT_EQ(model->getVars().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 1);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	ScheduledModel scheduledModel = ScheduledModel(model->getVars());
	for (ModEquation eq : model->getEquations())
		scheduledModel.addUpdate(eq);
	for (ModBltBlock bltBlock : model->getBltBlocks())
		scheduledModel.addUpdate(bltBlock);

	auto assignModel = addBltBlocks(scheduledModel);
	if (!assignModel)
	{
		outs() << assignModel.takeError();
		FAIL();
	}

	EXPECT_EQ(assignModel->getVars().size(), 1);
	EXPECT_EQ(assignModel->getUpdates().size(), 1);
	EXPECT_TRUE(holds_alternative<ModBltBlock>(assignModel->getUpdates()[0]));

	ModBltBlock bltBlock = get<ModBltBlock>(assignModel->getUpdates()[0]);
	EXPECT_EQ(bltBlock.getEquations().size(), 1);
	EXPECT_EQ(bltBlock.getResidual().size(), 1);
	EXPECT_EQ(bltBlock.getJacobian().size(), 1);
	EXPECT_EQ(bltBlock.getJacobian().front().size(), 1);
}
