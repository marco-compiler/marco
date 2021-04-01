#include <fcntl.h>
#include <llvm/ADT/iterator_range.h>
#include <modelica/mlirlowerer/passes/matching/Edge.h>
#include <modelica/mlirlowerer/passes/matching/Flow.h>

using namespace modelica::codegen::model;

Flow::Flow(IndexSet set, IndexSet mapped, Edge& edge, bool isForward)
		: edge(&edge),
			set(std::move(set)),
			mappedFlow(std::move(mapped)),
			isForward(isForward)
{
}

Edge& Flow::getEdge()
{
	return *edge;
}

const Edge& Flow::getEdge() const
{
	return *edge;
}

const Variable& Flow::getVariable() const
{
	return getEdge().getVariable();
}

const Equation& Flow::getEquation() const
{
	return edge->getEquation();
}

const modelica::IndexSet& Flow::getSet() const
{
	return set;
}

bool Flow::empty() const
{
	return set.empty();
}

size_t Flow::size() const
{
	return set.size();
}

const modelica::IndexSet& Flow::getMappedSet() const
{
	return mappedFlow;
}

bool Flow::isForwardEdge() const
{
	return isForward;
}

void Flow::addFLowAtEnd(IndexSet& set)
{
	if (isForwardEdge())
		edge->getSet().unite(set);
	else
		edge->getSet().remove(set);
}

modelica::IndexSet Flow::inverseMap(const IndexSet& set) const
{
	if (isForwardEdge())
		return edge->invertMap(set);

	return edge->map(set);
}

modelica::IndexSet Flow::applyAndInvert(IndexSet set)
{
	if (isForwardEdge())
		set = inverseMap(set);

	addFLowAtEnd(set);

	if (!isForwardEdge())
		set = inverseMap(set);
	return set;
}

Flow Flow::forwardedge(Edge& edge, IndexSet set)
{
	IndexSet mapped = edge.map(set);
	return Flow(std::move(set), std::move(mapped), edge, true);
}

Flow Flow::backedge(Edge& edge, IndexSet set)
{
	IndexSet mapped = edge.invertMap(set);
	return Flow(std::move(mapped), std::move(set), edge, false);
}

bool Flow::compare(const Flow& l, const Flow& r, const MatchingGraph& g)
{
	if (l.isForwardEdge())
	{
		auto lDeg = g.outDegree(l.getEquation());
		auto rDeg = g.outDegree(r.getEquation());
		if (lDeg != rDeg)
			return lDeg < rDeg;
	}
	else
	{
		auto lDeg = g.outDegree(l.getVariable());
		auto rDeg = g.outDegree(r.getVariable());
		if (lDeg != rDeg)
			return lDeg < rDeg;
	}

	return l.size() < r.size();
};

FlowCandidates::FlowCandidates(llvm::SmallVector<Flow, 2> c, const MatchingGraph& g)
		: choices(std::move(c))
{
	assert(std::find_if(choices.begin(), choices.end(),
											[](const Flow& flow) {
												return flow.empty();
											}) == choices.end());

	sort(choices, [&](const auto& l, const auto& r) {
		return Flow::compare(l, r, g);
	});
}

bool FlowCandidates::empty() const
{
	return choices.empty();
}

FlowCandidates::const_iterator FlowCandidates::begin() const
{
	return choices.begin();
}

FlowCandidates::const_iterator FlowCandidates::end() const
{
	return choices.end();
}

Flow& FlowCandidates::getCurrent()
{
	assert(!choices.empty());
	return choices.back();
}

const Flow& FlowCandidates::getCurrent() const
{
	assert(!choices.empty());
	return choices.back();
}

const Variable& FlowCandidates::getCurrentVariable() const
{
	return getCurrent().getEdge().getVariable();
}

void FlowCandidates::pop()
{
	assert(choices.begin() != choices.end());
	auto last = choices.end();
	last--;
	choices.erase(last, choices.end());
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

size_t AugmentingPath::size() const
{
	return frontier.size();
}

FlowCandidates& AugmentingPath::getCurrentCandidates()
{
	return frontier.back();
}

const FlowCandidates& AugmentingPath::getCurrentCandidates() const
{
	return frontier.back();
}

FlowCandidates AugmentingPath::getBestCandidate() const
{
	if (!getCurrentFlow().isForwardEdge())
		return getForwardMatchable();

	return getBackwardMatchable();
}

Flow& AugmentingPath::getCurrentFlow()
{
	return getCurrentCandidates().getCurrent();
}

const Flow& AugmentingPath::getCurrentFlow() const
{
	return getCurrentCandidates().getCurrent();
}

FlowCandidates AugmentingPath::selectStartingEdge() const
{
	llvm::SmallVector<Flow, 2> possibleStarts;

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

static modelica::IndexSet possibleForwardFlow(
		const Flow& backEdge, const Edge& forwadEdge, const MatchingGraph& graph)
{
	assert(!backEdge.isForwardEdge());
	auto direct = backEdge.getSet();
	direct.intersecate(forwadEdge.getEquation().getInductions());
	return direct;
}

modelica::IndexSet AugmentingPath::possibleBackwardFlow(const Edge& backEdge) const
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

void AugmentingPath::apply()
{
	assert(valid());

	auto alreadyMatchedVars = graph.getMatchedSet(getCurrentFlow().getVariable());
	auto set = getCurrentFlow().getMappedSet();
	set.remove(alreadyMatchedVars);

	auto reverseRange = llvm::make_range(std::rbegin(frontier), std::rend(frontier));
	for (auto& edge : reverseRange)
	{
		Flow& flow = edge.getCurrent();
		set = flow.applyAndInvert(set);
	}
}

FlowCandidates AugmentingPath::getBackwardMatchable() const
{
	assert(getCurrentFlow().isForwardEdge());
	llvm::SmallVector<Flow, 2> undoingMatch;

	auto connectedEdges = graph.arcsOf(getCurrentFlow().getVariable());
	for (Edge& edge : connectedEdges)
	{
		if (&edge == &getCurrentFlow().getEdge())
			continue;
		auto backFlow = possibleBackwardFlow(edge);
		if (!backFlow.empty())
			undoingMatch.emplace_back(Flow::backedge(edge, std::move(backFlow)));
	}

	return FlowCandidates(undoingMatch, graph);
}

FlowCandidates AugmentingPath::getForwardMatchable() const
{
	assert(!getCurrentFlow().isForwardEdge());
	llvm::SmallVector<Flow, 2> directMatch;

	auto connectedEdges = graph.arcsOf(getCurrentFlow().getEquation());
	for (Edge& edge : connectedEdges)
	{
		if (&edge == &getCurrentFlow().getEdge())
			continue;
		auto possibleFlow = possibleForwardFlow(getCurrentFlow(), edge, graph);
		if (!possibleFlow.empty())
			directMatch.emplace_back(Flow::forwardedge(edge, std::move(possibleFlow)));
	}

	return FlowCandidates(directMatch, graph);
}

