#include "modelica/matching/Flow.hpp"

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
FlowCandidates AugmentingPath::getForwardMatchable() const
{
	assert(!getCurrentFlow().isForwardEdge());
	SmallVector<Flow, 2> directMatch;

	for (Edge& edge : graph.arcsOf(getCurrentFlow().getEquation()))
	{
		auto direct = getCurrentFlow().getSet();
		auto alreadyUsed = graph.getMatchedSet(edge.getEquation());
		direct.remove(alreadyUsed);

		if (!direct.empty())
			directMatch.emplace_back(Flow::forwardedge(edge, move(direct)));
	}

	return directMatch;
}

FlowCandidates AugmentingPath::getBackwardMatchable() const
{
	assert(getCurrentFlow().isForwardEdge());
	SmallVector<Flow, 2> undoingMatch;

	for (Edge& edge : graph.arcsOf(getCurrentFlow().getVariable()))
	{
		auto alreadyAssigned = edge.map(edge.getSet());
		auto possibleFlow = getCurrentFlow().getMappedSet();
		alreadyAssigned.remove(possibleFlow);

		if (!alreadyAssigned.empty())
			undoingMatch.emplace_back(Flow::backedge(edge, move(alreadyAssigned)));
	}

	return undoingMatch;
}

FlowCandidates AugmentingPath::getBestCandidate() const
{
	if (!getCurrentFlow().isForwardEdge())
		return getForwardMatchable();

	return getBackwardMatchable();
}

AugmentingPath::AugmentingPath(MatchingGraph& graph): frontier(), graph(graph)
{
	auto startingCandidates = selectStartingEdge();

	if (startingCandidates.empty())
		return;

	frontier.push_back(move(startingCandidates));

	while (!frontier.empty() && !valid())
	{
		if (getCurrentCandidates().empty())
		{
			frontier.erase(frontier.end() - 1);
			getCurrentCandidates().pop();
			continue;
		}

		frontier.push_back(getBestCandidate());
	}
}

void AugmentingPath::apply()
{
	assert(valid());

	auto alreadyMatchedVars = graph.getMatchedSet(getCurrentFlow().getVariable());
	auto set = getCurrentFlow().getMappedSet();
	set.remove(alreadyMatchedVars);
	for (auto edge = frontier.rbegin(); edge != frontier.rend(); edge++)
	{
		set = edge->getCurrent().inverseMap(set);
		edge->getCurrent().addFLowAtEnd(set);
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
