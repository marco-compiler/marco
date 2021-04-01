#pragma once

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <modelica/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <modelica/utils/GraphIterator.hpp>
#include <type_traits>

namespace modelica::codegen::model
{
	/**
	 * a scc is a vector of vertex descriptors that are all reachable from each
	 * other in the graph. It allows get a range to iterate over the vertex
	 * properties of the indexes in this scc.
	 */
	template<typename Graph>
	class Scc
	{
		public:
		using VertexDesc = typename Graph::VertexDesc;
		using Vector = llvm::SmallVector<VertexDesc, 3>;
		using Iter = typename Vector::iterator;
		using ConstIter = typename Vector::const_iterator;

		Scc(Vector indexes): indexes(std::move(indexes)) {}
		Scc() = default;

		[[nodiscard]] auto range(const Graph& toIterateOver) const
		{
			using VertexDesc = typename std::remove_reference<decltype(toIterateOver[0])>::type;
			using Iterator = GraphIterator<const Graph, ConstIter, const VertexDesc>;

			auto begin = Iterator(toIterateOver, indexes.begin());
			auto end = Iterator(toIterateOver, indexes.end());
			return llvm::make_range<Iterator>(begin, end);
		}

		void push_back(VertexDesc index) { indexes.push_back(index); }

		[[nodiscard]] size_t size() const { return indexes.size(); }

		[[nodiscard]] VertexDesc operator[](size_t index) const
		{
			return indexes[index];
		}

		[[nodiscard]] auto range(Graph& toIterateOver)
		{
			using VertexDesc =
					typename std::remove_reference<decltype(toIterateOver[0])>::type;
			using Iterator = GraphIterator<Graph, Iter, VertexDesc>;

			auto begin = Iterator(toIterateOver, indexes.begin());
			auto end = Iterator(toIterateOver, indexes.end());
			return llvm::make_range<Iterator>(begin, end);
		}

		[[nodiscard]] auto begin() const { return indexes.begin(); }
		[[nodiscard]] auto begin() { return indexes.begin(); }
		[[nodiscard]] auto end() const { return indexes.end(); }
		[[nodiscard]] auto end() { return indexes.end(); }

		private:
		Vector indexes;
	};

	/**
	 * A scc lookup is a map like object that given a vertex returns the scc it
	 * belongs to or given a scc index returns the scc object at that index.
	 */
	template<typename Graph>
	class SccLookup
	{
		public:
		using VertDesc = typename Graph::VertexDesc;
		using SCC = Scc<Graph>;
		using SccVector = llvm::SmallVector<SCC, 3>;
		using VertToSCC = llvm::SmallVector<size_t, 3>;

		SccLookup(const Graph& graph): allScc(0), vertToScc(graph.count())
		{
			auto componentsCount = boost::strong_components(
					graph.getImpl(),
					boost::make_iterator_property_map(
							vertToScc.begin(),
							boost::get(boost::vertex_index, graph.getImpl())));

			allScc = SccVector(componentsCount);
			for (auto i : irange(vertToScc.size()))
			{
				const auto scc = vertToScc[i];
				allScc[scc].push_back(i);
			}
		}

		[[nodiscard]] size_t count() const { return allScc.size(); }

		[[nodiscard]] auto begin() const { return allScc.begin(); }
		[[nodiscard]] auto end() const { return allScc.end(); }
		[[nodiscard]] auto begin() { return allScc.begin(); }
		[[nodiscard]] auto end() { return allScc.end(); }
		[[nodiscard]] const SCC& sccOf(VertDesc vertexIndex) const
		{
			return allScc[vertToScc[vertexIndex]];
		}
		[[nodiscard]] const SCC& operator[](size_t index) const
		{
			return allScc[index];
		}

		private:
		SccVector allScc;
		VertToSCC vertToScc;
	};
}	 // namespace modelica
