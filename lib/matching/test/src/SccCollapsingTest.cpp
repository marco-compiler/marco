#include "gtest/gtest.h"
#include <iterator>

#include "llvm/InitializePasses.h"
#include "marco/matching/KhanAdjacentAlgorithm.hpp"
#include "marco/matching/SVarDependencyGraph.hpp"
#include "marco/matching/Schedule.hpp"
#include "marco/matching/VVarDependencyGraph.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
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
