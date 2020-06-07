#include "gtest/gtest.h"
#include <iterator>

#include "llvm/InitializePasses.h"
#include "modelica/matching/KhanAdjacentAlgorithm.hpp"
#include "modelica/matching/SVarDependencyGraph.hpp"
#include "modelica/matching/Schedule.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"
#include "modelica/model/ModExp.hpp"
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

TEST(SCCcollapsingTest, EquationShoudlBeNormalizable)
{
	auto model = makeModel();
	auto norm = model.getEquation(1).normalized();
	EXPECT_EQ(
			model.getEquation(1).getInductions(), MultiDimInterval({ { 2, 4 } }));
	auto acc = AccessToVar::fromExp(model.getEquation(0).getLeft());
	EXPECT_TRUE(acc.getAccess().isIdentity());
}
