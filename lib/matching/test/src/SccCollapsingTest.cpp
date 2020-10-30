#include "gtest/gtest.h"
#include <iterator>

#include "llvm/InitializePasses.h"
#include "modelica/matching/KhanAdjacentAlgorithm.hpp"
#include "modelica/matching/SVarDependencyGraph.hpp"
#include "modelica/matching/Schedule.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModType.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/Interval.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

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
TEST(SCCcollapsingTest, ThreeDepthNormalization)
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

	auto e = eq.normalized();
	auto acc = AccessToVar::fromExp(e.getLeft());
	EXPECT_TRUE(acc.getAccess().isIdentity());
	auto acc2 = AccessToVar::fromExp(e.getRight());
	EXPECT_TRUE(acc2.getAccess().isIdentity());
	EXPECT_EQ(
			e.getInductions(), MultiDimInterval({ { 0, 1 }, { 0, 4 }, { 0, 4 } }));
}

TEST(SCCcollapsingTest, EquationShoudlBeNormalizable)
{
	auto model = makeModel();
	auto norm = model.getEquation(1).normalized();
	EXPECT_EQ(
			model.getEquation(1).getInductions(), MultiDimInterval({ { 2, 4 } }));
	auto acc = AccessToVar::fromExp(model.getEquation(0).getLeft());
	EXPECT_TRUE(acc.getAccess().isIdentity());
}
