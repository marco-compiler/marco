#include "gtest/gtest.h"

#include "modelica/matching/VVarDependencyGraph.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

TEST(VVarDependency, countTest)
{
	EntryModel model;
	model.emplaceVar(
			"leftVar",
			ModExp(ModConst(0, 1, 2, 3), ModType(BultinModTypes::INT, 2, 2)));
	model.emplaceEquation(
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModConst(3),
			{ InductionVar(1, 3) });

	VVarDependencyGraph graph(model);
	EXPECT_EQ(graph.count(), 1);
}

TEST(VVarDependency, paritionTest)
{
	EntryModel model;
	model.emplaceVar(
			"leftVar", ModExp(ModConst(0, 1), ModType(BultinModTypes::INT, 2)));
	model.emplaceVar(
			"rightVar", ModExp(ModConst(0, 1), ModType(BultinModTypes::INT, 2)));
	model.emplaceEquation(
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModConst(3),
			{ InductionVar(0, 2) });

	model.emplaceEquation(
			ModExp::at(
					ModExp("rightVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst(0))),
			{ InductionVar(0, 2) });

	VVarDependencyGraph graph(model);
	auto scc = graph.getSCC();
	EXPECT_EQ(scc.count(), 2);
}
