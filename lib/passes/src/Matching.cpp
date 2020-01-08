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

static Optional<Edge> toEdge(
		const Model& model,
		const ModEquation& eq,
		const ModExp& use,
		size_t useIndex)
{
	assert(
			!use.isReference() ||
			model.getVar(use.getReference()).getInit().getModType().isScalar());

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

class EdgeFlow
{
	public:
	EdgeFlow(Edge& edge, IndexSet set, bool isForward)
			: edge(&edge),
				set(move(set)),
				mappedFlow(edge.map(set)),
				isForward(isForward)
	{
	}

	EdgeFlow(Edge& edge, bool isForward): edge(&edge), isForward(isForward) {}

	static EdgeFlow backedge(Edge& edge, IndexSet set)
	{
		return EdgeFlow(move(set), edge, false);
	}
	static EdgeFlow forwardedge(Edge& edge, IndexSet set)
	{
		return EdgeFlow(move(set), edge, true);
	}

	[[nodiscard]] const Edge& getEdge() const { return *edge; }
	[[nodiscard]] const ModEquation& getEquation() const
	{
		return edge->getEquation();
	}
	[[nodiscard]] const IndexSet& getSet() const { return set; }
	[[nodiscard]] const IndexSet& getMappedSet() const { return mappedFlow; }
	[[nodiscard]] size_t size() const { return set.size(); }

	[[nodiscard]] static bool compare(const EdgeFlow& l, const EdgeFlow& r)
	{
		return l.size() < r.size();
	};
	[[nodiscard]] bool isForwardEdge() const { return isForward; }
	void addFLowAtEnd(IndexSet& set)
	{
		if (isForwardEdge())
			edge->getSet().unite(set);
		else
			edge->getSet().remove(set);
	}
	[[nodiscard]] IndexSet inverseMap(const IndexSet& set) const
	{
		if (isForwardEdge())
			return edge->map(set);
		return edge->invertMap(set);
	}

	private:
	EdgeFlow(IndexSet set, Edge& edge, bool isForward)
			: edge(&edge),
				set(edge.invertMap(set)),
				mappedFlow(move(set)),
				isForward(isForward)
	{
	}
	Edge* edge;
	IndexSet set;
	IndexSet mappedFlow;
	bool isForward;
};

static IndexSet getUnmatchedSet(
		const MatchingGraph& graph, const ModEquation& equation)
{
	IndexSet matched;
	const auto unite = [&matched](const auto& edge) {
		matched.unite(edge.getSet());
	};
	graph.forAllConnected(equation, unite);

	auto set = equation.toIndexSet();
	set.remove(matched);
	return set;
}

class FlowCandidates
{
	public:
	[[nodiscard]] auto begin() const { return choises.begin(); }
	[[nodiscard]] auto begin() { return choises.begin(); }
	[[nodiscard]] auto end() const { return choises.end(); }
	[[nodiscard]] auto end() { return choises.end(); }
	FlowCandidates(SmallVector<EdgeFlow, 2> c)
			: choises(move(c)), current(std::begin(choises))
	{
		sort();
	}

	void sort() { llvm::sort(begin(), end(), EdgeFlow::compare); }
	[[nodiscard]] bool empty() const { return choises.empty(); }
	[[nodiscard]] bool allVisited() const { return current == choises.end(); }
	void next()
	{
		do
			current++;
		while (current < end() && current->getSet().empty());
	}
	[[nodiscard]] EdgeFlow& getCurrent() { return *current; }
	[[nodiscard]] const EdgeFlow& getCurrent() const { return *current; }
	[[nodiscard]] const ModVariable& getCurrentVariable() const
	{
		return current->getEdge().getVariable();
	}

	private:
	SmallVector<EdgeFlow, 2> choises;
	SmallVector<EdgeFlow, 2>::iterator current;
};

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

static FlowCandidates selectStartingEdge(MatchingGraph& graph)
{
	SmallVector<EdgeFlow, 2> unmatchedEqus;

	for (const auto& eq : graph.getModel())
	{
		graph.forAllConnected(eq, [&](Edge& e) {
			unmatchedEqus.emplace_back(
					EdgeFlow::forwardedge(e, getUnmatchedSet(graph, eq)));
		});
	}

	return unmatchedEqus;
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
	if ((candidates.size() % 2) != 1)
		return false;

	auto& current = candidates.back();
	auto set = graph.getMatchedSet(current.getCurrentVariable());
	set.remove(current.getCurrent().getMappedSet());
	return !set.empty();
}

static SmallVector<EdgeFlow, 2> findAugmentingPath(MatchingGraph& graph)
{
	SmallVector<EdgeFlow, 2> path;
	SmallVector<FlowCandidates, 2> candidates{ selectStartingEdge(graph) };

	while (!isAugmenthingPath(graph, candidates) && !candidates.empty())
	{
		auto& currentCandidates = candidates.back();
		if (!currentCandidates.allVisited())
		{
			candidates.push_back(getBestCandidate(graph, currentCandidates));
			currentCandidates.next();
		}

		while (candidates.back().allVisited())
			candidates.erase(candidates.end() - 1);
	}

	for (const auto& cand : candidates)
		path.push_back(move(cand.getCurrent()));

	return path;
}

static bool updatePath(MatchingGraph& graph, SmallVector<EdgeFlow, 2> flow)
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

static bool tryExpand(MatchingGraph& graph)
{
	return updatePath(graph, findAugmentingPath(graph));
}

Matching modelica::match(const Model& model)
{
	MatchingGraph graph(model);

	while (tryExpand(graph))
		;

	Matching toReturn;

	for (auto& edge : graph)
		if (!edge.getSet().empty())
			toReturn.push_back(move(edge.getMatchedEquation()));

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
				 << ";";
		});

		equationIndex++;
	}

	OS << "}\n";
}
