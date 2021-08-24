#include "marco/matching/SCCDependencyGraph.hpp"

#include <boost/range/iterator_range_core.hpp>
#include <iterator>

#include "marco/matching/VVarDependencyGraph.hpp"

using namespace boost;
using namespace marco;
using namespace llvm;
using namespace std;

SCCDependencyGraph::SCCDependencyGraph(VVarDependencyGraph& originalGraph)
		: sccLookup(originalGraph), originalGraph(originalGraph)
{
	map<const Scc*, VertexDesc> insertedVertex;
	for (const auto& scc : sccLookup)
		insertedVertex[&scc] = add_vertex(&scc, graph);

	auto edgeIterator = make_iterator_range(edges(originalGraph.getImpl()));
	for (const auto& edge : edgeIterator)
	{
		size_t targetVertex = target(edge, originalGraph.getImpl());
		size_t sourceVertex = source(edge, originalGraph.getImpl());

		if (targetVertex == sourceVertex)
			continue;

		const auto& targetScc = sccLookup.sccOf(targetVertex);
		const auto& sourceScc = sccLookup.sccOf(sourceVertex);

		add_edge(insertedVertex[&sourceScc], insertedVertex[&targetScc], graph);
	}
}

SmallVector<const Scc<VVarDependencyGraph>*, 0>
SCCDependencyGraph::topologicalSort() const
{
	SmallVector<size_t, 0> sorted(sccLookup.count(), 0);
	SmallVector<const Scc*, 0> out(sccLookup.count(), nullptr);
	topological_sort(graph, sorted.begin());

	for (auto i : irange(sorted.size()))
		out[i] = &sccLookup[sorted[i]];

	return out;
}
