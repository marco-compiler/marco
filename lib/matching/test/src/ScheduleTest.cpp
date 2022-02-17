#include "gtest/gtest.h"

#include "marco/matching/Matching.hpp"
#include "marco/matching/SccCollapsing.hpp"
#include "marco/matching/Schedule.hpp"
#include "marco/model/Assigment.hpp"
#include "marco/model/ModBltBlock.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModParser.hpp"

using namespace std;
using namespace llvm;
using namespace marco;

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
	/* Tested Model:
		model Sched4
			int x;
			int y;
			int[4] w;
			int z;
		equation
			for i in 3:3 loop
				w[i] - w[i+1] = y - w[i-2];
			end for;
			y = w[2] + x;
			for j in 1:1 loop
				w[j] + w[j+1] = x;
			end for;
			w[4] = 7 - w[3];
			w[3] = z - y;
			x = 5;
			w[1] - w[2] = 3;
		end Sched4;
	*/

	const string stringModel =
			"init "
			"w = INT[4]call fill INT[4](INT[1]{0}) "
			"x = INT[1]{0} "
			"y = INT[1]{0} "
			"z = INT[1]{0} "
			"template "
			"eq_1 INT[1](at INT[4]w, INT[1]{3}) = INT[1](+ INT[1](* INT[1](at "
			"INT[4]w, INT[1]{2}), INT[1]{-1}), INT[1]{7}) "
			"eq_0 INT[1]y = INT[1](+ INT[1](at INT[4]w, INT[1]{1}), INT[1]x) "
			"eq_5 INT[1](+ INT[1](at INT[4]w, INT[1](+ INT[1](ind INT[1]{0}), "
			"INT[1]{-1})), INT[1](* INT[1](at INT[4]w, INT[1](ind INT[1]{0})), "
			"INT[1]{-1})) = INT[1](+ INT[1]y, INT[1](* INT[1](at INT[4]w, INT[1](+ "
			"INT[1](ind INT[1]{0}), INT[1]{-3})), INT[1]{-1})) "
			"eq_6 INT[1](+ INT[1](at INT[4]w, INT[1](+ INT[1](ind INT[1]{0}), "
			"INT[1]{-1})), INT[1](at INT[4]w, INT[1](ind INT[1]{0}))) = INT[1]x "
			"eq_2 INT[1](at INT[4]w, INT[1]{2}) = INT[1](+ INT[1]z, INT[1](* "
			"INT[1]y, INT[1]{-1})) "
			"eq_3 INT[1]x = INT[1]{5} "
			"eq_4 INT[1](+ INT[1](at INT[4]w, INT[1]{0}), INT[1](* INT[1](at "
			"INT[4]w, INT[1]{1}), INT[1]{-1})) = INT[1]{3} "
			"update "
			"template eq_0 "
			"template eq_1 "
			"template eq_2 "
			"template eq_3 "
			"template eq_4 "
			"for [3,4]template eq_5 "
			"for [1,2]template eq_6 ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
		FAIL();

	EXPECT_EQ(model->getVars().size(), 4);
	EXPECT_EQ(model->getEquations().size(), 7);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto matchedModel = match(move(*model), 1000);
	if (!matchedModel)
		FAIL();

	EXPECT_EQ(matchedModel->getVars().size(), 4);
	EXPECT_EQ(matchedModel->getEquations().size(), 7);
	EXPECT_EQ(matchedModel->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*matchedModel), 1000);
	if (!collapsedModel)
		FAIL();

	EXPECT_EQ(collapsedModel->getVars().size(), 4);
	EXPECT_EQ(collapsedModel->getEquations().size(), 3);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 2);

	auto scheduledModel = schedule(move(*collapsedModel));

	EXPECT_EQ(scheduledModel.getVars().size(), 4);
	EXPECT_EQ(scheduledModel.getUpdates().size(), 5);
	EXPECT_TRUE(holds_alternative<ModEquation>(scheduledModel.getUpdates()[0]));
	EXPECT_TRUE(holds_alternative<ModBltBlock>(scheduledModel.getUpdates()[1]));
	EXPECT_TRUE(holds_alternative<ModEquation>(scheduledModel.getUpdates()[2]));
	EXPECT_TRUE(holds_alternative<ModBltBlock>(scheduledModel.getUpdates()[3]));
	EXPECT_TRUE(holds_alternative<ModEquation>(scheduledModel.getUpdates()[4]));
}

TEST(ScheduleTest, BltBlockAndVectorEquation)
{
	/* Tested Model:
		model Sched5
			int[2] z;
			int[10] x;
			int[5] y;
		equation
			y[1] + y[2] = x[3];
			for i in 1:5 loop
				z[i] = x[i+5] - y[1];
			end for;
			y[2] = x[7] + y[1];
			for j in 1:10 loop
				j + 5 = x[j];
			end for;
		end Sched5;
	*/

	const string stringModel =
			"init "
			"x = INT[10]call fill INT[10](INT[1]{0}) "
			"y = INT[5]call fill INT[5](INT[1]{0}) "
			"z = INT[2]call fill INT[2](INT[1]{0}) "
			"template "
			"eq_1 INT[1](at INT[5]y, INT[1]{1}) = INT[1](+ INT[1](at INT[10]x, "
			"INT[1]{6}), INT[1](at INT[5]y, INT[1]{0})) "
			"eq_0 INT[1](+ INT[1](at INT[5]y, INT[1]{0}), INT[1](at INT[5]y, "
			"INT[1]{1})) = INT[1](at INT[10]x, INT[1]{2}) "
			"eq_3 INT[1](+ INT[1](ind INT[1]{0}), INT[1]{5}) = INT[1](at INT[10]x, "
			"INT[1](+ INT[1](ind INT[1]{0}), INT[1]{-1})) "
			"eq_2 INT[1](at INT[2]z, INT[1](+ INT[1](ind INT[1]{0}), INT[1]{-1})) = "
			"INT[1](+ INT[1](at INT[10]x, INT[1](+ INT[1](ind INT[1]{0}), "
			"INT[1]{4})), INT[1](* INT[1](at INT[5]y, INT[1]{0}), INT[1]{-1})) "
			"update "
			"template eq_0 "
			"template eq_1 "
			"for [1,6]template eq_2 "
			"for [1,11]template eq_3 ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
		FAIL();

	EXPECT_EQ(model->getVars().size(), 3);
	EXPECT_EQ(model->getEquations().size(), 4);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto matchedModel = match(move(*model), 1000);
	if (!matchedModel)
		FAIL();

	EXPECT_EQ(matchedModel->getVars().size(), 3);
	EXPECT_EQ(matchedModel->getEquations().size(), 4);
	EXPECT_EQ(matchedModel->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*matchedModel), 1000);
	if (!collapsedModel)
		FAIL();

	EXPECT_EQ(collapsedModel->getVars().size(), 3);
	EXPECT_EQ(collapsedModel->getEquations().size(), 2);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 1);

	auto scheduledModel = schedule(move(*collapsedModel));

	EXPECT_EQ(scheduledModel.getVars().size(), 3);
	EXPECT_EQ(scheduledModel.getUpdates().size(), 3);
	EXPECT_TRUE(holds_alternative<ModEquation>(scheduledModel.getUpdates()[0]));
	EXPECT_EQ(
			get<ModEquation>(scheduledModel.getUpdates()[0]).getInductions().size(),
			10);
	EXPECT_TRUE(holds_alternative<ModBltBlock>(scheduledModel.getUpdates()[1]));
	EXPECT_TRUE(holds_alternative<ModEquation>(scheduledModel.getUpdates()[2]));
	EXPECT_EQ(
			get<ModEquation>(scheduledModel.getUpdates()[2]).getInductions().size(),
			5);
}
