#include "gtest/gtest.h"

#include "modelica/matching/SccCollapsing.hpp"
#include "modelica/matching/Schedule.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModBltBlock.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModParser.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

TEST(ScheduleTest, SimpleScheduling)
{
	/* Tested Model:
		model Sched1
			int[3] x;
			int[3] y;
		equation
			y[3] + 7 = x[2];
			x[1] = 3 * y[2];
			y[2] = x[3] + 2;
			x[1] + x[2] = 0;
			y[1] = -y[3];
			x[3] = 7;
		end Sched1;
	*/

	const string stringModel =
			"init "
			"x = INT[3]call fill INT[3](INT[1]{0}) "
			"y = INT[3]call fill INT[3](INT[1]{0}) "
			"template "
			"eq_0m0 INT[1](+ INT[1](at INT[3]y, INT[1]{2}), INT[1]{7}) = INT[1](at "
			"INT[3]x, INT[1]{1}) "
			"eq_1m1 INT[1](at INT[3]x, INT[1]{0}) = INT[1](* INT[1]{3}, INT[1](at "
			"INT[3]y, INT[1]{1})) "
			"eq_2m2 INT[1](at INT[3]y, INT[1]{1}) = INT[1](+ INT[1](at INT[3]x, "
			"INT[1]{2}), INT[1]{2}) "
			"eq_3m3 INT[1](+ INT[1](at INT[3]x, INT[1]{0}), INT[1](at INT[3]x, "
			"INT[1]{1})) = INT[1]{0} "
			"eq_4m4 INT[1](at INT[3]y, INT[1]{0}) = INT[1](* INT[1](at INT[3]y, "
			"INT[1]{2}), INT[1]{-1}) "
			"eq_5m5 INT[1](at INT[3]x, INT[1]{2}) = INT[1]{7} "
			"update "
			"for [0,1]template eq_0m0 matched [0,0] "
			"for [0,1]template eq_1m1 matched [0] "
			"for [0,1]template eq_2m2 matched [0] "
			"for [0,1]template eq_3m3 matched [0,1] "
			"for [0,1]template eq_4m4 matched [0] "
			"for [0,1]template eq_5m5 matched [0] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
		FAIL();

	EXPECT_EQ(model->getVars().size(), 2);
	EXPECT_EQ(model->getEquations().size(), 6);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*model), 1000);
	if (!collapsedModel)
		FAIL();

	EXPECT_EQ(collapsedModel->getVars().size(), 2);
	EXPECT_EQ(collapsedModel->getEquations().size(), 6);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 0);

	auto scheduledModel = schedule(move(*collapsedModel));

	EXPECT_EQ(scheduledModel.getVars().size(), 2);
	EXPECT_EQ(scheduledModel.getUpdates().size(), 6);

	ModExp varX = ModExp("x", ModType(BultinModTypes::INT, 3));
	ModExp varY = ModExp("y", ModType(BultinModTypes::INT, 3));
	SmallVector<ModEquation, 5> result = {
		ModEquation(ModExp::at(varX, ModConst(2)), ModConst(7)),
		ModEquation(
				ModExp::at(varY, ModConst(1)),
				ModExp::add(ModExp::at(varX, ModConst(2)), ModConst(2))),
		ModEquation(
				ModExp::at(varX, ModConst(0)),
				ModExp::multiply(ModConst(3), ModExp::at(varY, ModConst(1)))),
		ModEquation(
				ModExp::add(
						ModExp::at(varX, ModConst(0)), ModExp::at(varX, ModConst(1))),
				ModConst(0)),
		ModEquation(
				ModExp::add(ModExp::at(varY, ModConst(2)), ModConst(7)),
				ModExp::at(varX, ModConst(1))),
		ModEquation(
				ModExp::at(varY, ModConst(0)),
				ModExp::multiply(ModExp::at(varY, ModConst(2)), ModConst(-1)))
	};

	for (auto i : irange(result.size()))
	{
		EXPECT_EQ(
				get<ModEquation>(scheduledModel.getUpdates()[i]).getLeft(),
				result[i].getLeft());
		EXPECT_EQ(
				get<ModEquation>(scheduledModel.getUpdates()[i]).getRight(),
				result[i].getRight());
	}
}

TEST(ScheduleTest, EquationBeforeBltBlock)
{
	/* Tested Model:
		model Sched2
			int[3] x;
		equation
			x[1] + x[2] = 2;
			4 = x[3];
			x[1] - x[2] = x[3];
		end Sched2;
	*/

	const string stringModel =
			"init "
			"x = INT[3]call fill INT[3](INT[1]{0}) "
			"template "
			"eq_0m0 INT[1](+ INT[1](at INT[3]x, INT[1]{0}), INT[1](at INT[3]x, "
			"INT[1]{1})) = INT[1]{2} "
			"eq_1m1 INT[1]{4} = INT[1](at INT[3]x, INT[1]{2}) "
			"eq_2m2 INT[1](+ INT[1](at INT[3]x, INT[1]{0}), INT[1](* INT[1](at "
			"INT[3]x, INT[1]{1}), INT[1]{-1})) = INT[1](at INT[3]x, INT[1]{2}) "
			"update "
			"for [0,1]template eq_0m0 matched [0,0] "
			"for [0,1]template eq_1m1 matched [1] "
			"for [0,1]template eq_2m2 matched [0,1,0] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
		FAIL();

	EXPECT_EQ(model->getVars().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 3);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*model), 1000);
	if (!collapsedModel)
		FAIL();

	EXPECT_EQ(collapsedModel->getVars().size(), 1);
	EXPECT_EQ(collapsedModel->getEquations().size(), 1);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 1);

	auto scheduledModel = schedule(move(*collapsedModel));

	EXPECT_EQ(scheduledModel.getVars().size(), 1);
	EXPECT_EQ(scheduledModel.getUpdates().size(), 2);
	EXPECT_TRUE(holds_alternative<ModEquation>(scheduledModel.getUpdates()[0]));
	EXPECT_TRUE(holds_alternative<ModBltBlock>(scheduledModel.getUpdates()[1]));
}

TEST(ScheduleTest, BltBlockBeforeEquation)
{
	/* Tested Model:
		model Sched3
			int[3] x;
		equation
			x[1] + x[2] = 2;
			x[2] = x[3];
			x[1] - x[2] = 4;
		end Sched3;
	*/

	const string stringModel =
			"init "
			"x = INT[3]call fill INT[3](INT[1]{0}) "
			"template "
			"eq_0m0 INT[1](+ INT[1](at INT[3]x, INT[1]{0}), INT[1](at INT[3]x, "
			"INT[1]{1})) = INT[1]{2} "
			"eq_1m1 INT[1](at INT[3]x, INT[1]{1}) = INT[1](at INT[3]x, INT[1]{2}) "
			"eq_2m2 INT[1](+ INT[1](at INT[3]x, INT[1]{0}), INT[1](* INT[1](at "
			"INT[3]x, INT[1]{1}), INT[1]{-1})) = INT[1]{4} "
			"update "
			"for [0,1]template eq_0m0 matched [0,1] "
			"for [0,1]template eq_1m1 matched [1] "
			"for [0,1]template eq_2m2 matched [0,0] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
		FAIL();

	EXPECT_EQ(model->getVars().size(), 1);
	EXPECT_EQ(model->getEquations().size(), 3);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*model), 1000);
	if (!collapsedModel)
		FAIL();

	EXPECT_EQ(collapsedModel->getVars().size(), 1);
	EXPECT_EQ(collapsedModel->getEquations().size(), 1);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 1);

	auto scheduledModel = schedule(move(*collapsedModel));

	EXPECT_EQ(scheduledModel.getVars().size(), 1);
	EXPECT_EQ(scheduledModel.getUpdates().size(), 2);
	EXPECT_TRUE(holds_alternative<ModBltBlock>(scheduledModel.getUpdates()[0]));
	EXPECT_TRUE(holds_alternative<ModEquation>(scheduledModel.getUpdates()[1]));
}

TEST(ScheduleTest, MultipleBltBlocksAndEquations)
{
	// TODO
	EXPECT_EQ(true, true);
}
