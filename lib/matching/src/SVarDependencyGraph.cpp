#include "modelica/matching/SVarDependencyGraph.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/detail/adjacency_list.hpp>
#include <cstddef>
#include <map>

#include "llvm/ADT/SmallVector.h"
#include "modelica/matching/MatchedEquationLookup.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"

using namespace modelica;
using namespace llvm;
using namespace boost;
using namespace std;

void SVarDepencyGraph::insertNode(LookUp& LookUp, size_t vertexIndex)
{
	const IndexesOfEquation& vertex = collapsedGraph[vertexIndex];
	const auto& interval = vertex.getInterval();
	for (auto indicies : interval.contentRange())
	{
		auto vertexIndex =
				add_vertex(SingleEquationReference(vertex, indicies), graph);
		LookUp[&vertex].push_back(vertexIndex);
	}
}

void SVarDepencyGraph::insertEdge(
		const LookUp& lookUp, const VVarDependencyGraph::EdgeDesc& edge)
{
	const auto& sourceVertex = source(edge, collImpl());
	const auto& targetVertex = target(edge, collImpl());
	const IndexesOfEquation& targetIndexes = *collImpl()[targetVertex];
	const IndexesOfEquation& sourceIndexes = *collImpl()[sourceVertex];
	const auto& edgeInfo = collImpl()[edge];
}

void SVarDepencyGraph::insertEdges(const LookUp& lookUp, size_t vertexIndex)
{
	for (const auto& edge : outEdgesRange(vertexIndex, collImpl()))
		insertEdge(lookUp, edge);
}

SVarDepencyGraph::SVarDepencyGraph(
		const VVarDependencyGraph& collapsedGraph, const VVarScc& scc)
		: scc(scc), collapsedGraph(collapsedGraph)
{
	LookUp vertexesLookUp;
	for (const auto& vertex : scc)
		insertNode(vertexesLookUp, vertex);

	for (size_t vertex : scc)
		insertEdges(vertexesLookUp, vertex);
}
