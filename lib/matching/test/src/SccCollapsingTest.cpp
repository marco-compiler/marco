#include "gtest/gtest.h"
#include <iterator>

#include "llvm/InitializePasses.h"
#include "marco/matching/KhanAdjacentAlgorithm.hpp"
#include "marco/matching/SVarDependencyGraph.hpp"
#include "marco/matching/SccCollapsing.hpp"
#include "marco/matching/Schedule.hpp"
#include "marco/matching/VVarDependencyGraph.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModParser.hpp"
#include "marco/model/ModType.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/Interval.hpp"

using namespace std;
using namespace llvm;
using namespace marco;

static auto makeModel()
{
	Model model;
	model.emplaceVar(
			"leftVar", ModExp(ModConst(0, 1), ModType(BultinModTypes::INT, 2)));
	model.emplaceVar(
			"rightVar", ModExp(ModConst(0, 1), ModType(BultinModTypes::INT, 2)));
	model.emplaceEquation(
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModConst(3),
			"",
			{ { 0, 2 } });

	model.emplaceEquation(
			ModExp::at(
					ModExp("rightVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0)) + ModExp(ModConst(-2))),
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0)) + ModExp(ModConst(-2))),
			"",
			{ { 2, 4 } });
	return model;
}

TEST(SccCollapsingTest, ThreeDepthNormalization)
{
	auto exp = ModExp::at(
			ModExp::at(
					ModExp::at(
							ModExp("rightVar", ModType(BultinModTypes::INT, 4, 4, 4)),
							ModExp::induction(ModConst(0)) + ModExp(ModConst(-1))),
					ModExp::induction(ModConst(1)) + ModExp(ModConst(-1))),
			ModExp::induction(ModConst(2)) + ModExp(ModConst(-1)));
	ModEquation eq(
			exp, exp, "", MultiDimInterval({ { 1, 2 }, { 1, 5 }, { 1, 5 } }));

	auto norm = eq.normalized();
	if (!norm)
		FAIL();
	auto e = *norm;

	auto acc = AccessToVar::fromExp(e.getLeft());
	EXPECT_TRUE(acc.getAccess().isIdentity());
	auto acc2 = AccessToVar::fromExp(e.getRight());
	EXPECT_TRUE(acc2.getAccess().isIdentity());
	EXPECT_EQ(
			e.getInductions(), MultiDimInterval({ { 0, 1 }, { 0, 4 }, { 0, 4 } }));
}

TEST(SccCollapsingTest, EquationShouldBeNormalizable)
{
	auto model = makeModel();
	auto norm = model.getEquation(1).normalized();
	if (!norm)
		FAIL();
	EXPECT_EQ(
			model.getEquation(1).getInductions(), MultiDimInterval({ { 2, 4 } }));
	auto acc = AccessToVar::fromExp(model.getEquation(0).getLeft());
	EXPECT_TRUE(acc.getAccess().isIdentity());
}

TEST(SccCollapsingTest, SccsWithVectors)
{
	const string stringModel =
			"init "
			"x = FLOAT[3]call fill FLOAT[3](INT[1]{0}) "
			"y = FLOAT[2]call fill FLOAT[2](INT[1]{0}) "
			"template "
			"eq_0m0 FLOAT[1](at FLOAT[3]x, INT[1](+ INT[1](ind INT[1]{0}), "
			"INT[1]{-1})) = FLOAT[1](* FLOAT[1](at FLOAT[2]y, INT[1](+ INT[1](ind "
			"INT[1]{0}), INT[1]{-1})), FLOAT[1]{2.000000e+00}) "
			"eq_1m1 FLOAT[1](at FLOAT[3]x, INT[1](+ INT[1](ind INT[1]{0}), "
			"INT[1]{-1})) = INT[1]{2} "
			"eq_2m2 FLOAT[1](at FLOAT[3]x, INT[1](ind INT[1]{0})) = FLOAT[1](* "
			"FLOAT[1](at FLOAT[2]y, INT[1](+ INT[1](ind INT[1]{0}), INT[1]{-1})), "
			"FLOAT[1]{3.000000e+00}) "
			"eq_3m3 INT[1]{4} = FLOAT[1](* FLOAT[1](at FLOAT[2]y, INT[1](+ "
			"INT[1](ind INT[1]{0}), INT[1]{-1})), FLOAT[1]{3.000000e+00}) "
			"update "
			"for [1,3]template eq_0m0 matched [0] "
			"for [3,4]template eq_1m1 matched [0] "
			"for [1,2]template eq_2m2 matched [1,0] "
			"for [2,3]template eq_3m3 matched [1,0] ";

	ModParser parser(stringModel);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	EXPECT_EQ(model->getVars().size(), 2);
	EXPECT_EQ(model->getEquations().size(), 4);
	EXPECT_EQ(model->getBltBlocks().size(), 0);

	auto collapsedModel = solveScc(move(*model), 1000);
	if (!collapsedModel)
	{
		outs() << collapsedModel.takeError();
		FAIL();
	}

	EXPECT_EQ(collapsedModel->getVars().size(), 2);
	EXPECT_EQ(collapsedModel->getEquations().size(), 5);
	EXPECT_EQ(collapsedModel->getBltBlocks().size(), 0);
}

TEST(SccCollapsingTest, CyclesWithScalarsInBltBlock)
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
	}
}

TEST(SccCollapsingTest, CyclesWithVectorsInBltBlock)
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
	}
}

TEST(SccCollapsingTest, CycleMoreThanTwoEquations)
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
}
