#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/detail/adjacency_list.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <cstddef>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/InitializePasses.h>
#include <map>
#include <marco/mlirlowerer/passes/matching/MatchedEquationLookup.h>
#include <marco/mlirlowerer/passes/matching/SVarDependencyGraph.h>
#include <marco/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <optional>

using namespace marco::codegen::model;
using namespace llvm;
using namespace boost;
using namespace std;

void SVarDependencyGraph::insertNode(LookUp& lookUp, size_t vertexIndex)
{
	const IndexesOfEquation& vertex = collapsedGraph[vertexIndex];
	
	for (size_t i : irange(vertex.size()))
	{
		const MultiDimInterval& interval = vertex.getEquations()[i].getInductions();
		for (auto eqInds : interval.contentRange())
		{
			auto indicies = vertex.getEqToVars()[i].map(eqInds);
			size_t vertexIndex = add_vertex(SingleEquationReference(vertex, indicies), graph);
			lookUp[&vertex][vertex.getVariables()[i].indexOfElement(indicies)] = vertexIndex;
		}
	}
}

static Optional<size_t> indexOfScalarVar(
		ArrayRef<size_t> access,
		const IndexesOfEquation& var,
		const SVarDependencyGraph::LookUp& lookUp)
{
	auto v = lookUp.find(&var);

	if (v == lookUp.end())
		return {};

	for (const Variable& variable : var.getVariables())
	{
		auto toReturn = v->second.find(variable.indexOfElement(access));
		if (toReturn != v->second.end())
			return toReturn->second;
	}
	return {};
}

void SVarDependencyGraph::insertEdge(
		const LookUp& lookUp, const VVarDependencyGraph::EdgeDesc& edge)
{
	size_t sourceVertex = source(edge, collImpl());
	size_t targetVertex = target(edge, collImpl());
	const IndexesOfEquation& targetNode = *collImpl()[targetVertex];

	if (lookUp.find(&targetNode) == lookUp.end())
		return;

	const IndexesOfEquation& sourceNode = *collImpl()[sourceVertex];
	const VectorAccess& varAccess = collImpl()[edge];

	for (size_t i : irange(targetNode.size()))
	{
		VectorAccess dependencies = targetNode.getVarToEqs()[i] * varAccess;

		for (const auto& indices : targetNode.getIntervals()[i].contentRange())
		{
			auto sourceIndex = indexOfScalarVar(indices, sourceNode, lookUp);

			auto scalarVarInduction = dependencies.map(indices);
			auto targetIndex = indexOfScalarVar(scalarVarInduction, targetNode, lookUp);

			if (!sourceIndex || !targetIndex)
				continue;

			add_edge(*sourceIndex, *targetIndex, graph);
		}
	}
}

void SVarDependencyGraph::insertEdges(const LookUp& lookUp, size_t vertexIndex)
{
	for (const auto& edge : outEdgesRange(vertexIndex, collImpl()))
		insertEdge(lookUp, edge);
}

SVarDependencyGraph::SVarDependencyGraph(
		const VVarDependencyGraph& collapsedGraph, const VVarScc& scc)
		: scc(scc), collapsedGraph(collapsedGraph)
{
	LookUp vertexesLookUp;

	for (const auto& vertex : scc)
		insertNode(vertexesLookUp, vertex);

	for (size_t vertex : scc)
		insertEdges(vertexesLookUp, vertex);
}
