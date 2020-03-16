#pragma once

#include <boost/graph/adjacency_list.hpp>

#include "llvm/ADT/SmallVector.h"
#include "modelica/matching/MatchedEquationLookup.hpp"
#include "modelica/matching/SccLookup.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"
namespace modelica
{
	class SingleEquationReference
	{
		public:
		SingleEquationReference(
				const IndexesOfEquation& vertex, llvm::SmallVector<size_t, 3> indexes)
				: vertex(&vertex), indexes(std::move(indexes))
		{
		}

		SingleEquationReference() = default;

		[[nodiscard]] const auto& getCollapsedVertex()
		{
			return vertex->getEquation();
		}

		private:
		const IndexesOfEquation* vertex;
		llvm::SmallVector<size_t, 3> indexes;
	};

	class SVarDepencyGraph
	{
		public:
		using SVarGraph = boost::adjacency_list<
				boost::vecS,
				boost::vecS,
				boost::directedS,
				SingleEquationReference>;

		using VertexIndex =
				boost::property_map<SVarGraph, boost::vertex_index_t>::type::value_type;

		using VVarVertexDesc = boost::graph_traits<SVarGraph>::vertex_descriptor;

		using VVarScc = Scc<VVarDependencyGraph::VVarVertexDesc>;

		SVarDepencyGraph(
				const VVarDependencyGraph& collapsedGraph, const VVarScc& scc);

		[[nodiscard]] const VVarScc& getScc() const { return scc; }
		[[nodiscard]] size_t count() const { return boost::num_vertices(graph); }

		private:
		const VVarScc& scc;
		const VVarDependencyGraph& collapsedGraph;
		SVarGraph graph;
	};
}	 // namespace modelica
