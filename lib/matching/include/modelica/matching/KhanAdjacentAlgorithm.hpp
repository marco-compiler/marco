#pragma once

#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <cstddef>
#include <limits>
#include <set>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "modelica/utils/IRange.hpp"
namespace modelica
{
	enum class khanNextPreferred
	{
		cannotBeOptimized,
		forwardPreferred,
		bothPreferred,
		backwardPreferred
	};
	template<typename Graph>
	void initKhanAlgorithm(
			Graph& graph,
			llvm::SmallVector<size_t, 0>& requisitesCount,
			std::set<size_t>& schedulableSet)
	{
		using namespace boost;
		using namespace std;
		using namespace llvm;
		auto verts = make_iterator_range(vertices(graph));
		for (size_t vertex : verts)
		{
			auto outEdges = make_iterator_range(out_edges(vertex, graph));
			for (const auto& outEdge : outEdges)
			{
				auto tVert = target(outEdge, graph);
				requisitesCount[tVert]++;
			}
		}

		for (size_t i : irange<size_t>(requisitesCount.size()))
			if (requisitesCount[i] == 0)
				schedulableSet.insert(i);
	}

	template<typename Graph, typename Iterator>
	void khanUpdate(
			Graph& graph,
			Iterator& iterator,
			llvm::SmallVector<size_t, 0>& requisitesCount,
			std::set<size_t>& schedulableSet,
			size_t toSchedule)
	{
		schedulableSet.erase(toSchedule);
		auto outEdges = make_iterator_range(out_edges(toSchedule, graph));
		for (const auto& outEdge : outEdges)
		{
			auto tVert = target(outEdge, graph);
			requisitesCount[tVert]--;
			if (requisitesCount[tVert] == 0)
				schedulableSet.insert(tVert);
		}
		*(iterator++) = toSchedule;
	}

	template<typename Graph>
	size_t khanSelectBest(
			Graph& graph,
			llvm::ArrayRef<size_t> requisitesCount,
			std::set<size_t>& schedulableSet,
			khanNextPreferred& schedDir,
			size_t lastSched)
	{
		// the starting node is selected at random
		if (schedDir == khanNextPreferred::cannotBeOptimized)
		{
			schedDir = khanNextPreferred::bothPreferred;
			return *schedulableSet.begin();
		}

		// if we already scheduled two nodes from the
		// same variable backward, or we scheduled one,
		// then we must try schedule the next backward
		if (schedDir == khanNextPreferred::bothPreferred ||
				schedDir == khanNextPreferred::backwardPreferred)
		{
			size_t previous = lastSched - 1;
			if (lastSched != 0 && requisitesCount[previous] == 0)
			{
				schedDir = khanNextPreferred::backwardPreferred;
				return previous;
			}
		}

		// if we already scheduled two nodes from the
		// same variable forward, or we scheduled one,
		// then we must try schedule the next forward
		if (schedDir == khanNextPreferred::bothPreferred ||
				schedDir == khanNextPreferred::forwardPreferred)
		{
			size_t next = lastSched + 1;
			if (next != requisitesCount.size() && requisitesCount[next] == 0)
			{
				schedDir = khanNextPreferred::forwardPreferred;
				return next;
			}
		}

		// schedule one at random
		schedDir = khanNextPreferred::bothPreferred;
		return *schedulableSet.begin();
	}

	template<typename Graph, typename OIterator>
	void khanAdjacentAlgorithm(Graph& graph, OIterator iterator)
	{
		using namespace boost;
		using namespace std;
		using namespace llvm;

		const size_t vertexCount = num_vertices(graph);
		if (vertexCount == 0)
			return;

		SmallVector<size_t, 0> requisitesCount(vertexCount, 0);
		set<size_t> schedulableSet;
		initKhanAlgorithm(graph, requisitesCount, schedulableSet);
		size_t lastScheduled = numeric_limits<size_t>::max();
		khanNextPreferred schedulingDirection(khanNextPreferred::cannotBeOptimized);

		if (schedulableSet.empty())
			assert(false && "graph was not a dag");

		while (!schedulableSet.empty())
		{
			lastScheduled = khanSelectBest(
					graph,
					requisitesCount,
					schedulableSet,
					schedulingDirection,
					lastScheduled);
			khanUpdate(
					graph, iterator, requisitesCount, schedulableSet, lastScheduled);
		}
	}
}	 // namespace modelica
