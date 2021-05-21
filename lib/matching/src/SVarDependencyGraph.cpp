#include "marco/matching/SVarDependencyGraph.hpp"

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
#include "marco/matching/MatchedEquationLookup.hpp"
#include "marco/matching/VVarDependencyGraph.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/model/VectorAccess.hpp"

using namespace marco;
using namespace llvm;
using namespace boost;
using namespace std;

void SVarDepencyGraph::insertNode(LookUp& LookUp, size_t vertexIndex)
{
	const IndexesOfEquation& vertex = collapsedGraph[vertexIndex];
	if (vertex.isEquation())
	{
		const ModEquation& eq = vertex.getEquation();
		const MultiDimInterval& interval = eq.getInductions();
		for (auto eqInds : interval.contentRange())
		{
			auto indicies = vertex.getEqToVar().map(eqInds);
			auto vertexIndex =
					add_vertex(SingleEquationReference(vertex, indicies), graph);
			LookUp[&vertex][vertex.getVariable().indexOfElement(indicies)] =
					vertexIndex;
		}
	}
	else	// TODO: Check if ModEquation/ModBltBlock is correctly differentiated
	{
		assert(false && "To be checked");
		const ModBltBlock& bltBlock = vertex.getBltBlock();
		for (auto i : irange(vertex.size()))
		{
			const ModEquation& eq = bltBlock.getEquation(i);
			const MultiDimInterval& interval = eq.getInductions();
			for (auto eqInds : interval.contentRange())
			{
				auto indicies = vertex.getEqToVars()[i].map(eqInds);
				auto vertexIndex =
						add_vertex(SingleEquationReference(vertex, indicies), graph);
				LookUp[&vertex][vertex.getVariables()[i]->indexOfElement(indicies)] =
						vertexIndex;
			}
		}
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

	if (var.isEquation())
	{
		auto toReturn = v->second.find(var.getVariable().indexOfElement(access));
		if (toReturn == v->second.end())
			return {};
		return toReturn->second;
	}
	else	// TODO: Check if ModEquation/ModBltBlock is correctly differentiated
	{
		assert(false && "To be checked");
		for (auto& variable : var.getVariables())
		{
			auto toReturn = v->second.find(variable->indexOfElement(access));
			if (toReturn != v->second.end())
				return toReturn->second;
		}
		return {};
	}
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

	if (targetNode.isEquation())
	{
		VectorAccess dependencies = targetNode.getVarToEq() * varAccess;

		for (const auto& indices : targetNode.getInterval().contentRange())
		{
			auto sourceIndex = indexOfScalarVar(indices, sourceNode, lookUp);

			auto scalarVarInduction = dependencies.map(indices);
			auto targetIndex =
					indexOfScalarVar(scalarVarInduction, targetNode, lookUp);
			if (!sourceIndex || !targetIndex)
				continue;

			add_edge(*sourceIndex, *targetIndex, graph);
		}
	}
	else	// TODO: Check if ModEquation/ModBltBlock is correctly differentiated
	{
		assert(false && "To be checked");
		for (auto i : irange(targetNode.size()))
		{
			VectorAccess dependencies = targetNode.getVarToEqs()[i] * varAccess;

			for (const auto& indices : targetNode.getIntervals()[i].contentRange())
			{
				auto sourceIndex = indexOfScalarVar(indices, sourceNode, lookUp);

				auto scalarVarInduction = dependencies.map(indices);
				auto targetIndex =
						indexOfScalarVar(scalarVarInduction, targetNode, lookUp);
				if (!sourceIndex || !targetIndex)
					continue;

				add_edge(*sourceIndex, *targetIndex, graph);
			}
		}
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
	for (auto i : irange(vertex->size()))
	{
		OS << vertex->getVariables()[i]->getName();
		OS << "[";
		for (size_t index : indexes)
			OS << index << ",";
		OS << "]";
		if (i < vertex->size() - 1)
			OS << "&";
	}
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
