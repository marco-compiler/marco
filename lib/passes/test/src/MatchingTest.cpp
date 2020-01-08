#include "gtest/gtest.h"

#include "modelica/model/Assigment.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModType.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/passes/Matching.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

TEST(MatchingTest, graphInizializationTest)
{
	EntryModel model;
	model.emplaceVar(
			"leftVar",
			ModExp(ModConst<int>(0, 1, 2, 3), ModType(BultinModTypes::INT, 2, 2)));
	model.emplaceEquation(
			ModExp::at(
					ModExp("leftVar", ModType(BultinModTypes::INT, 2, 2)),
					ModExp::induction(ModConst<int>(0))),
			ModConst<int>(3),
			{ InductionVar(1, 3) });

	MatchingGraph graph(model);
	EXPECT_EQ(graph.variableCount(), 1);
	EXPECT_EQ(graph.equationCount(), 1);
	EXPECT_EQ(graph.edgesCount(), 1);
}
