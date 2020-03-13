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
