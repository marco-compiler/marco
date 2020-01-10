#include "modelica/passes/Matching.hpp"

#include <type_traits>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModMatchers.hpp"
#include "modelica/model/ModVariable.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

/***
 * Creates a edge from a equation and a variable usage.
 * The use can either be a direct reference access or a
 * nested at operation terminating into a reference.
 *
 */
static Optional<Edge> toEdge(
		const Model& model,
		const ModEquation& eq,
		const ModExp& use,
		size_t useIndex)
{
	if (use.isReference())
		return Edge(eq, model.getVar(use.getReference()), useIndex);

	if (!use.isOperation())
		assert(false && "unreachable");

	auto maybeAccess = toVectorAccess(use);
	if (!maybeAccess)
		return Optional<Edge>();

	const auto& var = model.getVar(maybeAccess->getName());
	return Edge(eq, var, useIndex, move(*maybeAccess));
}

void MatchingGraph::addEquation(const ModEquation& eq)
{
	ReferenceMatcher matcher;
	visit(eq.getLeft(), matcher);
	visit(eq.getRight(), matcher);
	size_t useIndex = 0;

	for (const auto& use : matcher)
		addVariableUsage(eq, *use, useIndex++);
}

void MatchingGraph::addVariableUsage(
		const ModEquation& eq, const ModExp& use, size_t index)
{
	auto maybeEdge = toEdge(model, eq, use, index);
	if (maybeEdge.hasValue())
		emplace_edge(move(*maybeEdge));
}

static FlowCandidates getInstantMatchable(
		MatchingGraph& graph, const EdgeFlow& arrivingFlow)
{
	assert(!arrivingFlow.isForwardEdge());
	SmallVector<EdgeFlow, 2> directMatch;
	const auto getAviableForwardFlow = [&](Edge& edge) {
		auto direct = arrivingFlow.getSet();
		auto alreadyUsed = graph.getMatchedSet(edge.getVariable());
		direct.remove(alreadyUsed);

		if (!direct.empty())
			directMatch.emplace_back(edge, move(direct), true);
	};
	graph.forAllConnected(arrivingFlow.getEquation(), getAviableForwardFlow);
	return directMatch;
}

static FlowCandidates getBackwardMatchable(
		MatchingGraph& graph, const EdgeFlow& arrivingFlow)
{
	assert(arrivingFlow.isForwardEdge());
	SmallVector<EdgeFlow, 2> undoingMatch;
	const auto getAviableBackwardFlow = [&](Edge& edge) {
		auto undirect = arrivingFlow.getMappedSet();
		auto notYetUsed = graph.getUnmatchedSet(edge.getVariable());
		undirect.remove(notYetUsed);

		if (!undirect.empty())
			undoingMatch.emplace_back(EdgeFlow::backedge(edge, move(undirect)));
	};

	graph.forAllConnected(arrivingFlow.getEquation(), getAviableBackwardFlow);
	return undoingMatch;
}

FlowCandidates MatchingGraph::selectStartingEdge()
{
	SmallVector<EdgeFlow, 2> possibleStarts;

	for (const auto& eq : getModel())
	{
		IndexSet eqUnmatched = getUnmatchedSet(eq);
		if (eqUnmatched.empty())
			continue;

		forAllConnected(eq, [&](Edge& e) {
			possibleStarts.emplace_back(EdgeFlow::forwardedge(e, eqUnmatched));
		});
	}

	return possibleStarts;
}

static FlowCandidates getBestCandidate(
		MatchingGraph& graph, FlowCandidates& current)
{
	if (!current.getCurrent().isForwardEdge())
		return getInstantMatchable(graph, current.getCurrent());

	return getBackwardMatchable(graph, current.getCurrent());
}

static bool isAugmenthingPath(
		const MatchingGraph& graph,
		const SmallVector<FlowCandidates, 2>& candidates)
{
	if (candidates.empty())
		return false;
	if ((candidates.size() % 2) != 1)
		return false;

	auto& current = candidates.back();
	auto set = current.getCurrent().getMappedSet();
	set.remove(graph.getMatchedSet(current.getCurrentVariable()));

	return !set.empty();
}

SmallVector<EdgeFlow, 2> MatchingGraph::findAugmentingPath()
{
	SmallVector<EdgeFlow, 2> path;
	SmallVector<FlowCandidates, 2> candidates{ selectStartingEdge() };
	if (candidates.front().allVisited())
		return path;

	while (!isAugmenthingPath(*this, candidates) && !candidates.empty())
	{
		auto& currentCandidates = candidates.back();
		if (!currentCandidates.allVisited())
		{
			candidates.push_back(getBestCandidate(*this, currentCandidates));
			currentCandidates.next();
		}

		while (candidates.back().allVisited())
			candidates.erase(candidates.end() - 1);
	}

	for (auto& cand : candidates)
		path.push_back(move(cand.getCurrent()));

	return path;
}

bool MatchingGraph::updatePath(SmallVector<EdgeFlow, 2> flow)
{
	if (flow.empty())
		return false;

	auto set = flow.back().getMappedSet();
	for (auto edge = flow.rbegin(); edge != flow.rend(); edge++)
	{
		edge->addFLowAtEnd(set);
		set = edge->inverseMap(set);
	}
	return true;
}

void MatchingGraph::match(int maxIterations)
{
	while (maxIterations-- != 0 && updatePath(findAugmentingPath()))
		;
}

Matching MatchingGraph::extractMatch()
{
	Matching toReturn;

	for (auto& edge : *this)
		if (!edge.getSet().empty())
			toReturn.push_back(move(edge.getMatchedEquation()));

	return toReturn;
}

Matching MatchingGraph::toMatch() const
{
	Matching toReturn;

	for (auto& edge : *this)
		if (!edge.getSet().empty())
			toReturn.push_back(edge.getMatchedEquation());

	return toReturn;
}

void MatchingGraph::dumpGraph(raw_ostream& OS) const
{
	OS << "digraph {\n";

	int equationIndex = 1;
	for (const ModEquation& eq : model)
	{
		forAllConnected(eq, [&](const Edge& edge) {
			OS << "Eq_" << equationIndex << " -> " << edge.getVariable().getName()
				 << "[label=\"";
			edge.getSet().dump(OS);
			OS << "\"]"
				 << ";\n";
		});

		equationIndex++;
	}

	OS << "}\n";
}
