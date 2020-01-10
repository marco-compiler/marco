#include "gtest/gtest.h"

#include "modelica/model/Assigment.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModParser.hpp"
#include "modelica/model/ModType.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/passes/Matching.hpp"
#include "modelica/utils/IndexSet.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

const string s = "init "
								 "varA = INT[10] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0} "
								 "varB = INT[10] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} "
								 "update "
								 "for [0,5] "
								 "INT[1] (at INT[10] varA, INT[1](ind INT[1]{0})) = INT[1] (+ "
								 "INT[1] (at INT[10] varB, INT[1](ind INT[1]{0})), INT[1]{1}) "
								 "for [0,5] "
								 "INT[1] (at INT[10] varA, INT[1](ind INT[1]{0})) = INT[1] (+ "
								 "INT[1] (at INT[10] varB, INT[1](ind INT[1]{0})), INT[1]{2}) ";

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

TEST(MatchingTest, singleMatch)
{
	ModParser parser(s);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	auto [vars, equs] = *model;
	EntryModel m(move(equs), move(vars));
	MatchingGraph graph(m);
	graph.match(1);
	auto results = graph.toMatch();
	EXPECT_EQ(results.size(), 1);

	EXPECT_EQ(graph.selectStartingEdge().getCurrent().getSet().size(), 5);
}

TEST(MatchingTest, simpleMatch)
{
	ModParser parser(s);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	auto [vars, equs] = *model;
	EntryModel m(move(equs), move(vars));
	MatchingGraph graph(m);
	graph.match(2);
	auto results = graph.extractMatch();
	EXPECT_EQ(results.size(), 2);
}

TEST(MatchingTest, overRunningMatch)
{
	ModParser parser(s);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	auto [vars, equs] = *model;
	EntryModel m(move(equs), move(vars));
	MatchingGraph graph(m);
	graph.match(4);
	auto results = graph.extractMatch();
	EXPECT_EQ(results.size(), 2);
}

TEST(MatchingTest, firstMatchingSize)
{
	ModParser parser(s);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	auto [vars, equs] = *model;
	EntryModel m(move(equs), move(vars));
	MatchingGraph graph(m);
	FlowCandidates res = graph.selectStartingEdge();
	EXPECT_EQ(res.getCurrent().getSet().size(), 5);
	EXPECT_EQ(res.getCurrent().getEdge().getSet(), IndexSet());
	EXPECT_EQ(res.getCurrent().isForwardEdge(), true);
}

TEST(MatchingTest, firstMatchingVectorConstruction)
{
	ModParser parser(s);

	auto model = parser.simulation();
	if (!model)
	{
		outs() << model.takeError();
		FAIL();
	}

	auto [vars, equs] = *model;
	EntryModel m(move(equs), move(vars));
	MatchingGraph graph(m);

	SmallVector<FlowCandidates, 2> candidates{ graph.selectStartingEdge() };
	EXPECT_EQ(candidates[0].getCurrent().getEdge().getSet(), IndexSet());
}
