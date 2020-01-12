#include "modelica/matching/Flow.hpp"

#include "llvm/ADT/iterator_range.h"
#include "modelica/matching/Edge.hpp"
#include "modelica/matching/Matching.hpp"
#include "modelica/utils/IndexSet.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

void FlowCandidates::dump(llvm::raw_ostream& OS) const
{
	OS << "CURRENT:";
	for (const auto& c : choises)
		c.dump(OS);
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

FlowCandidates::FlowCandidates(SmallVector<Flow, 2> c): choises(std::move(c))
{
	assert(find_if(choises.begin(), choises.end(), [](const Flow& flow) {
					 return flow.empty();
				 }) == choises.end());
	std::sort(begin(choises), end(choises), Flow::compare);
}

bool AugmentingPath::valid() const
{
	if (frontier.empty())
		return false;
	if (getCurrentCandidates().empty())
		return false;
	if ((frontier.size() % 2) != 1)
		return false;

	const auto& currentVar = getCurrentCandidates().getCurrentVariable();
	auto set = getCurrentFlow().getMappedSet();
	set.remove(graph.getMatchedSet(currentVar));

	return !set.empty();
}

FlowCandidates AugmentingPath::selectStartingEdge() const
{
	SmallVector<Flow, 2> possibleStarts;

	for (const auto& eq : graph.getModel())
	{
		IndexSet eqUnmatched = graph.getUnmatchedSet(eq);
		if (eqUnmatched.empty())
			continue;

		for (Edge& e : graph.arcsOf(eq))
			possibleStarts.emplace_back(Flow::forwardedge(e, eqUnmatched));
	}

	return possibleStarts;
}
static IndexSet possibleForwardFlow(
		const Flow& backEdge, const Edge& forwadEdge, const MatchingGraph& graph)
{
	assert(!backEdge.isForwardEdge());
	auto direct = backEdge.getSet();
	auto alreadyUsed = graph.getMatchedSet(forwadEdge.getEquation());
	direct.remove(alreadyUsed);
	return direct;
}

FlowCandidates AugmentingPath::getForwardMatchable() const
{
	assert(!getCurrentFlow().isForwardEdge());
	SmallVector<Flow, 2> directMatch;

	auto connectedEdges = graph.arcsOf(getCurrentFlow().getEquation());
	for (Edge& edge : connectedEdges)
	{
		auto possibleFlow = possibleForwardFlow(getCurrentFlow(), edge, graph);
		if (!possibleFlow.empty())
			directMatch.emplace_back(Flow::forwardedge(edge, move(possibleFlow)));
	}

	return directMatch;
}

static IndexSet possibleBackwardFlow(
		const Flow& forwardEdge, const Edge& backEdge)
{
	assert(forwardEdge.isForwardEdge());
	auto alreadyAssigned = backEdge.map(backEdge.getSet());
	auto possibleFlow = forwardEdge.getMappedSet();
	alreadyAssigned.remove(possibleFlow);
	return alreadyAssigned;
}

FlowCandidates AugmentingPath::getBackwardMatchable() const
{
	assert(getCurrentFlow().isForwardEdge());
	SmallVector<Flow, 2> undoingMatch;

	auto connectedEdges = graph.arcsOf(getCurrentFlow().getVariable());
	for (Edge& edge : connectedEdges)
	{
		auto backFlow = possibleBackwardFlow(getCurrentFlow(), edge);
		if (!backFlow.empty())
			undoingMatch.emplace_back(Flow::backedge(edge, move(backFlow)));
	}

	return undoingMatch;
}

FlowCandidates AugmentingPath::getBestCandidate() const
{
	if (!getCurrentFlow().isForwardEdge())
		return getForwardMatchable();

	return getBackwardMatchable();
}

AugmentingPath::AugmentingPath(MatchingGraph& graph)
		: graph(graph), frontier({ selectStartingEdge() })
{
	while (!valid())
	{
		// while the current siblings are
		// not empty keep exploring
		if (!getCurrentCandidates().empty())
		{
			frontier.push_back(getBestCandidate());
			continue;
		}

		// if they are empty remove the last group
		frontier.erase(frontier.end() - 1);

		// if the frontier is now empty we are done
		// there is no good path
		if (frontier.empty())
			return;

		// else remove one of the siblings
		getCurrentCandidates().pop();
	}
}

void AugmentingPath::apply()
{
	assert(valid());

	auto alreadyMatchedVars = graph.getMatchedSet(getCurrentFlow().getVariable());
	auto set = getCurrentFlow().getMappedSet();
	set.remove(alreadyMatchedVars);

	auto reverseRange = make_range(rbegin(frontier), rend(frontier));
	for (auto edge : reverseRange)
	{
		Flow& flow = edge.getCurrent();
		set = flow.inverseMap(set);
		flow.addFLowAtEnd(set);
	}
}

void AugmentingPath::dump(llvm::raw_ostream& OS) const
{
	OS << "valid path = " << (valid() ? "true" : "false");
	OS << '\n';
	for (const auto& e : frontier)
		e.dump(OS);
}
string AugmentingPath::toString() const
{
	string str;
	raw_string_ostream ss(str);
	dump(ss);
	ss.flush();
	return str;
}
