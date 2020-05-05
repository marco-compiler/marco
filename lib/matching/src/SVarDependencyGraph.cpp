#include "modelica/matching/SVarDependencyGraph.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/detail/adjacency_list.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <cstddef>
#include <map>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
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
	const auto& interval = vertex.getEquation().getInductions();
	for (auto eqInds : interval.contentRange())
	{
		auto indicies = vertex.getEqToVar().map(eqInds);
		auto vertexIndex =
				add_vertex(SingleEquationReference(vertex, indicies), graph);
		LookUp[&vertex][vertex.getVariable().indexOfElement(indicies)] =
				vertexIndex;
	}
}

static Optional<size_t> indexOfScalarVar(
		ArrayRef<size_t> access,
		const IndexesOfEquation& var,
		const SVarDepencyGraph::LookUp& lookUp)
{
	auto v = lookUp.find(&var);
	if (v == lookUp.end())
		return {};

	auto toReturn = v->second.find(var.getVariable().indexOfElement(access));
	if (toReturn == v->second.end())
		return {};
	return toReturn->second;
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
		auto sourceIndex = indexOfScalarVar(indecies, sourceNode, lookUp);

		auto scalarVarInduction = dependencies.map(indecies);
		auto targetIndex = indexOfScalarVar(scalarVarInduction, targetNode, lookUp);
		if (!sourceIndex || !targetIndex)
			continue;

		add_edge(*sourceIndex, *targetIndex, graph);
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

void SingleEquationReference::dump(llvm::raw_ostream& OS) const
{
	OS << vertex->getVariable().getName();
	OS << "[";
	for (size_t index : indexes)
		OS << index << ",";

	OS << "]";
}

void SVarDepencyGraph::dumpGraph(llvm::raw_ostream& OS) const
{
	OS << "digraph {";

	const auto& verts = make_iterator_range(vertices(graph));
	for (auto vertex : verts)
	{
		const auto& eqRef = graph[vertex];
		eqRef.dump(OS);
		OS << "[label=\"";
		eqRef.dump(OS);
		OS << "\"];\n";
	}

	const auto& edgs = make_iterator_range(edges(graph));
	for (const auto& edge : edgs)
	{
		const auto& sourceVertex = graph[source(edge, graph)];
		const auto& targetVertex = graph[target(edge, graph)];

		sourceVertex.dump(OS);
		OS << " -> ";
		targetVertex.dump(OS);
		OS << "\n";
	}

	OS << "};";
}
