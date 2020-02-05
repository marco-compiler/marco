#include "gtest/gtest.h"

#include "modelica/matching/Flow.hpp"
#include "modelica/matching/Matching.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModParser.hpp"
#include "modelica/model/ModType.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"
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
								 "INT[1] (at INT[10] varB, INT[1](+ INT[1](ind INT[1]{0}), "
								 "INT[1]{2})), INT[1]{2}) ";

TEST(MatchingTest, graphInizializationTest)
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
	EXPECT_EQ(graph.variableCount(), 2);
	EXPECT_EQ(graph.equationCount(), 2);
	EXPECT_EQ(graph.edgesCount(), 4);
	graph.match(1);
	EXPECT_EQ(graph.matchedEdgesCount(), 1);

	AugmentingPath path(graph);

	EXPECT_EQ(path.selectStartingEdge().getCurrent().getSet().size(), 5);
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
	EXPECT_EQ(graph.matchedEdgesCount(), 2);
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
	EXPECT_EQ(graph.matchedCount(), 10);
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
	AugmentingPath path(graph);
	FlowCandidates res = path.selectStartingEdge();
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
	AugmentingPath path(graph);

	auto candidates = path.selectStartingEdge();
	EXPECT_EQ(candidates.getCurrent().getEdge().getSet(), IndexSet());
}

TEST(MatchingTest, vectorAccessTest)
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

	SmallVector<VectorAccess, 2> access;
	for (const auto& edge : graph)
		access.push_back(edge.getVectorAccess());

	int count = 0;
	const string s = "varB";

	for (const auto& edge : graph)
	{
		VectorAccess acc = edge.getVectorAccess();
		VectorAccess copy(s, { SingleDimensionAccess::relative(2, 0) });
		if (acc == copy)
			count++;
	}

	EXPECT_EQ(count, 1);
}

TEST(MatchingTest, emptyGraph)
{
	EntryModel m({}, {});
	MatchingGraph graph(m);
	graph.match(4);
	EXPECT_EQ(graph.matchedEdgesCount(), 0);
}
