#include "modelica/matching/SVarDependencyGraph.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/detail/adjacency_list.hpp>
#include <cstddef>
#include <map>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/InitializePasses.h"
#include "modelica/matching/MatchedEquationLookup.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"

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

static size_t indexOfScalarVar(
		ArrayRef<size_t> access,
		const IndexesOfEquation& var,
		const SVarDepencyGraph::LookUp& lookUp)
{
	return lookUp.at(&var)[var.getVariable().indexOfElement(access)];
}

void SVarDepencyGraph::insertEdge(
		const LookUp& lookUp, const VVarDependencyGraph::EdgeDesc& edge)
{
	size_t sourceVertex = source(edge, collImpl());
	size_t targetVertex = target(edge, collImpl());
	const IndexesOfEquation& targetNode = *collImpl()[targetVertex];
	if (lookUp.find(&targetNode) == lookUp.end())
		return;
	const IndexesOfEquation& sourceNode = *collImpl()[sourceVertex];
	const VectorAccess& varAccess = collImpl()[edge];

	VectorAccess dependencies = targetNode.getVarToEq() * varAccess;

	for (const auto& indecies : targetNode.getInterval().contentRange())
	{
		size_t sourceIndex = indexOfScalarVar(indecies, sourceNode, lookUp);
		size_t targetIndex =
				indexOfScalarVar(dependencies.map(indecies), targetNode, lookUp);

		add_edge(sourceIndex, targetIndex, graph);
	}
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
