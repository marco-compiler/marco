#include "modelica/matching/Flow.hpp"

#include <fcntl.h>

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
	for (const auto& c : make_range(rbegin(choises), rend(choises)))
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

FlowCandidates::FlowCandidates(SmallVector<Flow, 2> c, const MatchingGraph& g)
		: choises(std::move(c))
{
	assert(find_if(choises.begin(), choises.end(), [](const Flow& flow) {
					 return flow.empty();
				 }) == choises.end());
	sort(choises, [&](const auto& l, const auto& r) {
		return Flow::compare(l, r, g);
	});
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

	return FlowCandidates(possibleStarts, graph);
}
static IndexSet possibleForwardFlow(
		const Flow& backEdge, const Edge& forwadEdge, const MatchingGraph& graph)
{
	assert(!backEdge.isForwardEdge());
	auto direct = backEdge.getSet();
	direct.intersecate(forwadEdge.getEquation().getInductions());
	return direct;
}

FlowCandidates AugmentingPath::getForwardMatchable() const
{
	assert(!getCurrentFlow().isForwardEdge());
	SmallVector<Flow, 2> directMatch;

	auto connectedEdges = graph.arcsOf(getCurrentFlow().getEquation());
	for (Edge& edge : connectedEdges)
	{
		if (&edge == &getCurrentFlow().getEdge())
			continue;
		auto possibleFlow = possibleForwardFlow(getCurrentFlow(), edge, graph);
		if (!possibleFlow.empty())
			directMatch.emplace_back(Flow::forwardedge(edge, move(possibleFlow)));
	}

	return FlowCandidates(directMatch, graph);
}

IndexSet AugmentingPath::possibleBackwardFlow(const Edge& backEdge) const
{
	const Flow& forwardEdge = getCurrentFlow();
	assert(forwardEdge.isForwardEdge());
	auto alreadyAssigned = backEdge.map(backEdge.getSet());
	auto possibleFlow = forwardEdge.getMappedSet();
	alreadyAssigned.intersecate(possibleFlow);

	for (const auto& siblingSet : frontier)
	{
		const auto currentEdge = siblingSet.getCurrent();
		if (&currentEdge.getEdge() != &backEdge)
			continue;
		if (currentEdge.isForwardEdge())
			continue;

		alreadyAssigned.remove(currentEdge.getMappedSet());
	}

	return alreadyAssigned;
}

FlowCandidates AugmentingPath::getBackwardMatchable() const
{
	assert(getCurrentFlow().isForwardEdge());
	SmallVector<Flow, 2> undoingMatch;

	auto connectedEdges = graph.arcsOf(getCurrentFlow().getVariable());
	for (Edge& edge : connectedEdges)
	{
		if (&edge == &getCurrentFlow().getEdge())
			continue;
		auto backFlow = possibleBackwardFlow(edge);
		if (!backFlow.empty())
			undoingMatch.emplace_back(Flow::backedge(edge, move(backFlow)));
	}

	return FlowCandidates(undoingMatch, graph);
}

FlowCandidates AugmentingPath::getBestCandidate() const
{
	if (!getCurrentFlow().isForwardEdge())
		return getForwardMatchable();

	return getBackwardMatchable();
}

AugmentingPath::AugmentingPath(MatchingGraph& graph, size_t maxDepth)
		: graph(graph), frontier({ selectStartingEdge() })
{
	while (!valid() && frontier.size() < maxDepth)
	{
		// while the current siblings are
		// not empty keep exploring
		if (!getCurrentCandidates().empty())
		{
			frontier.push_back(getBestCandidate());
			continue;
		}

		// if they are empty remove the last siblings group
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
  /* set adesso è il flusso da applicare */

  /* fai il giochino di andare avanti e indietro nel grafo per applicare il
   * flusso e nel contempo cancellarlo nei punti dove l'abbiamo assorbito */
	auto reverseRange = make_range(rbegin(frontier), rend(frontier));
	for (auto& edge : reverseRange)
	{
		Flow& flow = edge.getCurrent();
		set = flow.applyAndInvert(set);
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
void AugmentingPath::dumpGraph(
		raw_ostream& OS,
		bool displayEmptyEdges,
		bool displayMappings,
		bool displayOnlyMatchedCount,
		bool displayOtherOptions) const
{
	graph.dumpGraph(
			OS, displayEmptyEdges, displayMappings, displayOnlyMatchedCount, false);

	size_t candidateCount = 0;
	for (const auto& candidate : frontier)
	{
		size_t edgeIndex = 0;
		for (const auto& edge : candidate)
		{
			if (!displayOtherOptions && &edge != &candidate.getCurrent())
				continue;

			if (edge.isForwardEdge())
			{
				OS << "Eq_" << graph.indexOfEquation(edge.getEquation());
				OS << " -> " << edge.getVariable().getName();
			}
			else
			{
				OS << edge.getVariable().getName() << "->";
				OS << "Eq_" << graph.indexOfEquation(edge.getEquation());
			}
			OS << " [color=";
			OS << (&edge == &candidate.getCurrent() ? "gold" : "green");
			OS << ", label=" << candidateCount;
			OS << "];\n";
			edgeIndex++;
		}
		candidateCount++;
	}

	OS << "}\n";
}
