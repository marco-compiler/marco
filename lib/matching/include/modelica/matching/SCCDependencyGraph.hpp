#pragma once
#include <map>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/topological_sort.hpp"
#include "llvm/ADT/SmallVector.h"
#include "modelica/matching/SccLookup.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"

namespace modelica
{
	class SCCDependencyGraph
	{
		public:
		using GraphImp = boost::adjacency_list<
				boost::vecS,
				boost::vecS,
				boost::directedS,
				const Scc<size_t>*>;

		SCCDependencyGraph(SccLookup<size_t>& lookUp, VVarDependencyGraph& graph);

		[[nodiscard]] const VVarDependencyGraph& getVectorVarGraph() const
		{
			return originalGraph;
		}

		[[nodiscard]] llvm::SmallVector<const Scc<size_t>*, 0> topologicalSort()
				const
		{
			llvm::SmallVector<size_t, 0> sorted(sccLookup.count(), 0);
			llvm::SmallVector<const Scc<size_t>*, 0> out(sccLookup.count(), nullptr);
			boost::topological_sort(graph, sorted.rbegin());

			for (auto i : irange(sorted.size()))
				out[i] = &sccLookup[sorted[i]];

			return out;
		}

		private:
		GraphImp graph;
		SccLookup<size_t>& sccLookup;
		VVarDependencyGraph& originalGraph;
	};

}	 // namespace modelica
