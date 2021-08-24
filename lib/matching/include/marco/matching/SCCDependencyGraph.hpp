#pragma once
#include <boost/graph/graph_traits.hpp>
#include <map>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/topological_sort.hpp"
#include "llvm/ADT/SmallVector.h"
#include "marco/matching/SccLookup.hpp"
#include "marco/matching/VVarDependencyGraph.hpp"

namespace marco
{
	class SCCDependencyGraph
	{
		public:
		using Scc = SccLookup<VVarDependencyGraph>::SCC;

		using GraphImp = boost::
				adjacency_list<boost::vecS, boost::vecS, boost::directedS, const Scc*>;

		using VertexDesc = boost::graph_traits<GraphImp>::vertex_descriptor;

		SCCDependencyGraph(VVarDependencyGraph& graph);

		[[nodiscard]] const VVarDependencyGraph& getVectorVarGraph() const
		{
			return originalGraph;
		}

		[[nodiscard]] llvm::SmallVector<const Scc*, 0> topologicalSort() const;

		private:
		GraphImp graph;
		SccLookup<VVarDependencyGraph> sccLookup;
		VVarDependencyGraph& originalGraph;
	};

}	 // namespace marco
