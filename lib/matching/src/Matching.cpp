#include "modelica/matching/Matching.hpp"

#include <functional>
#include <type_traits>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/Edge.hpp"
#include "modelica/matching/Flow.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModMatchers.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

/***
 * Creates a edge from a equation and a variable usage.
 * The use can either be a direct reference access or a
 * nested at operation terminating into a reference.
 *
 */
static Edge toEdge(
		const Model& model,
		const ModEquation& eq,
		const ModExp& use,
		size_t useIndex)
{
	assert(isCanonicalVectorAccess(use));

	auto access = toVectorAccess(use);
	const auto& var = model.getVar(access.getName());
	return Edge(eq, var, useIndex, move(access));
}

void MatchingGraph::addEquation(const ModEquation& eq)
{
	ReferenceMatcher matcher;
	visit(eq.getLeft(), matcher);
	visit(eq.getRight(), matcher);
	size_t useIndex = 0;

	for (const auto& use : matcher)
	{
		if (!isCanonicalVectorAccess(*use))
			continue;

		size_t edgeIndex = edges.size();
		edges.push_back(toEdge(model, eq, *use, useIndex++));
		equationLookUp.insert({ &eq, edgeIndex });
		auto var = &(edges.back().getVariable());
		variableLookUp.insert({ var, edgeIndex });
	}
}

static FlowCandidates getForwardMatchable(
		MatchingGraph& graph, const Flow& arrivingFlow)
{
	assert(!arrivingFlow.isForwardEdge());
	SmallVector<Flow, 2> directMatch;
	const auto getAviableForwardFlow = [&](Edge& edge) {
		auto direct = arrivingFlow.getSet();
		auto alreadyUsed = graph.getMatchedSet(edge.getEquation());
		direct.remove(alreadyUsed);

		if (!direct.empty())
			directMatch.emplace_back(Flow::forwardedge(edge, move(direct)));
	};
	graph.forAllConnected(arrivingFlow.getEquation(), getAviableForwardFlow);
	return directMatch;
}

static FlowCandidates getBackwardMatchable(
		MatchingGraph& graph, const Flow& arrivingFlow)
{
	assert(arrivingFlow.isForwardEdge());
	SmallVector<Flow, 2> undoingMatch;
	const auto getAviableBackwardFlow = [&](Edge& edge) {
		auto alreadyAssigned = edge.map(edge.getSet());
		auto possibleFlow = arrivingFlow.getMappedSet();
		alreadyAssigned.remove(possibleFlow);

		if (!alreadyAssigned.empty())
			undoingMatch.emplace_back(Flow::backedge(edge, move(alreadyAssigned)));
	};

	graph.forAllConnected(
			arrivingFlow.getEdge().getVariable(), getAviableBackwardFlow);
	return undoingMatch;
}

FlowCandidates MatchingGraph::selectStartingEdge()
{
	SmallVector<Flow, 2> possibleStarts;

	for (const auto& eq : getModel())
	{
		IndexSet eqUnmatched = getUnmatchedSet(eq);
		if (eqUnmatched.empty())
			continue;

		forAllConnected(eq, [&](Edge& e) {
			possibleStarts.emplace_back(Flow::forwardedge(e, eqUnmatched));
		});
	}

	return possibleStarts;
}

static FlowCandidates getBestCandidate(
		MatchingGraph& graph, FlowCandidates& current)
{
	if (!current.getCurrent().isForwardEdge())
		return getForwardMatchable(graph, current.getCurrent());

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

SmallVector<FlowCandidates, 2> MatchingGraph::findAugmentingPath()
{
	SmallVector<FlowCandidates, 2> candidates{ selectStartingEdge() };
	while (!candidates.empty() && candidates.back().allVisited())
		candidates.erase(candidates.end() - 1);

	while (!isAugmenthingPath(*this, candidates) && !candidates.empty())
	{
		auto& currentCandidates = candidates.back();
		if (!currentCandidates.allVisited())
		{
			candidates.push_back(getBestCandidate(*this, currentCandidates));
			currentCandidates.next();
		}

		while (!candidates.empty() && candidates.back().allVisited())
			candidates.erase(candidates.end() - 1);
	}

	return candidates;
}

void MatchingGraph::match(int maxIterations)
{
	while (maxIterations-- != 0)
	{
		auto flow = findAugmentingPath();
		if (flow.empty())
			return;

		auto set = flow.back().getCurrent().getMappedSet();
		for (auto edge = flow.rbegin(); edge != flow.rend(); edge++)
		{
			edge->getCurrent().addFLowAtEnd(set);
			set = edge->getCurrent().inverseMap(set);
		}
	}
}

size_t MatchingGraph::matchedEdgesCount() const
{
	return count_if(
			begin(), end(), [](const auto& edge) { return !edge.empty(); });
}

void MatchingGraph::dumpGraph(raw_ostream& OS) const
{
	OS << "digraph {\n";

	int equationIndex = 1;
	for (const ModEquation& eq : model)
	{
		forAllConnected(eq, [&](const Edge& edge) {
			OS << "Eq_" << equationIndex << " -> " << edge.getVariable().getName();
			OS << "[label=\"";
			edge.getVectorAccess().dump(OS);
			OS << " ";
			edge.getSet().dump(OS);
			OS << " ";
			edge.getInvertedAccess().dump(OS);
			OS << "\"]"
				 << ";\n";
		});

		equationIndex++;
	}

	OS << "}\n";
}

void Edge::dump(llvm::raw_ostream& OS) const
{
	OS << "EDGE: Eq_" << equation << " TO " << variable->getName();
	OS << "\n";
	OS << "\tForward Map: ";
	vectorAccess.dump(OS);
	OS << " -> Backward Map: ";
	invertedAccess.dump(OS);
	OS << "\n\tCurrent Flow: ";
	set.dump(OS);
	OS << "\n";
}

string Edge::toString() const
{
	string str;
	raw_string_ostream ss(str);
	dump(ss);
	ss.flush();
	return str;
}

void MatchingGraph::dump(llvm::raw_ostream& OS) const
{
	for (const auto& edge : *this)
		edge.dump(OS);
}

void FlowCandidates::dump(llvm::raw_ostream& OS) const
{
	for (auto a = current; a < choises.size(); a++)
		choises[a].dump(OS);
}

string FlowCandidates::toString() const
{
	string str;
	raw_string_ostream ss(str);
	dump(ss);
	ss.flush();
	return str;
}

void Flow::dump(llvm::raw_ostream& OS) const
{
	edge->dump(OS);
	OS << "\t Forward=" << static_cast<int>(isForwardEdge()) << " Source Set:";
	set.dump(OS);
	OS << "-> Arriving Set:";
	mappedFlow.dump(OS);
	OS << "\n";
}
