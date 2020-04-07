#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/range/iterator_range.hpp>
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
	class KhanData
	{
		public:
		KhanData(const Graph& graph)
				: graph(graph), requisitesCount(boost::num_vertices(graph), 0)
		{
			countDependencies();
			collectSchedulableNodes();
		}

		void countDependency(size_t vertex)
		{
			auto outEdges = boost::make_iterator_range(out_edges(vertex, graph));

			for (const auto& outEdge : outEdges)
				requisitesCount[target(outEdge, graph)]++;
		}

		void countDependencies()
		{
			auto verts = boost::make_iterator_range(vertices(graph));
			for (size_t vertex : verts)
				countDependency(vertex);
		}

		void collectSchedulableNodes()
		{
			for (size_t i : irange<size_t>(requisitesCount.size()))
				if (requisitesCount[i] == 0)
					schedulableSet.insert(i);
		}

		void markAsScheduled(size_t node)
		{
			schedulableSet.erase(node);
			auto outEdges = make_iterator_range(out_edges(node, graph));
			for (const auto& outEdge : outEdges)
			{
				auto tVert = target(outEdge, graph);
				requisitesCount[tVert]--;
				if (requisitesCount[tVert] == 0)
					schedulableSet.insert(tVert);
			}
		}

		[[nodiscard]] size_t getLastScheduled() const { return lastScheduled; }

		template<typename OnElementScheduled, typename OnGroupFinished>
		void khanUpdate(
				OnElementScheduled& onElementScheduled,
				OnGroupFinished& onGroupFinish,
				size_t toSchedule)
		{
			markAsScheduled(toSchedule);
			onElementScheduled(toSchedule);

			if (groupFinished)
				onGroupFinish(lastScheduled, lastDirection);

			lastScheduled = toSchedule;
		}

		[[nodiscard]] bool canSchedule() const { return !schedulableSet.empty(); }

		size_t khanSelectBest()
		{
			if (!canSchedule())
				assert(false && "graph was not a dag");
			groupFinished = false;
			// the starting node is selected at random
			if (schedulingDirection == khanNextPreferred::cannotBeOptimized)
			{
				schedulingDirection = khanNextPreferred::bothPreferred;
				return *schedulableSet.begin();
			}

			// if we already scheduled two nodes from the
			// same variable backward, or we scheduled one,
			// then we must try schedule the next backward
			if (schedulingDirection == khanNextPreferred::bothPreferred ||
					schedulingDirection == khanNextPreferred::backwardPreferred)
			{
				size_t previous = lastScheduled - 1;
				if (lastScheduled != 0 && requisitesCount[previous] == 0)
				{
					schedulingDirection = khanNextPreferred::backwardPreferred;
					return previous;
				}
			}

			// if we already scheduled two nodes from the
			// same variable forward, or we scheduled one,
			// then we must try schedule the next forward
			if (schedulingDirection == khanNextPreferred::bothPreferred ||
					schedulingDirection == khanNextPreferred::forwardPreferred)
			{
				size_t next = lastScheduled + 1;
				if (next != requisitesCount.size() && requisitesCount[next] == 0)
				{
					schedulingDirection = khanNextPreferred::forwardPreferred;
					return next;
				}
			}

			// schedule one at random,break the group and begin a new group

			lastDirection = schedulingDirection;
			groupFinished = true;
			schedulingDirection = khanNextPreferred::bothPreferred;
			return *schedulableSet.begin();
		}

		[[nodiscard]] khanNextPreferred getLastDirection() const
		{
			return lastDirection;
		}

		private:
		khanNextPreferred schedulingDirection{
			khanNextPreferred::cannotBeOptimized
		};
		khanNextPreferred lastDirection{ khanNextPreferred::cannotBeOptimized };
		size_t lastScheduled{ std::numeric_limits<size_t>::max() };
		const Graph& graph;

		llvm::SmallVector<size_t, 0> requisitesCount;
		std::set<size_t> schedulableSet;
		bool groupFinished{ false };
	};

	template<typename Graph, typename OnElementScheduled, typename OnGroupFinish>
	void khanAdjacentAlgorithm(
			Graph& graph,
			OnElementScheduled onElementScheduled,
			OnGroupFinish onGroupFinish)
	{
		const size_t vertexCount = boost::num_vertices(graph);
		if (vertexCount == 0)
			return;

		KhanData s(graph);

		while (s.canSchedule())
			s.khanUpdate(onElementScheduled, onGroupFinish, s.khanSelectBest());
		onGroupFinish(s.getLastScheduled(), s.getLastDirection());
	}
}	 // namespace modelica
