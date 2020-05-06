#pragma once

#include <boost/graph/graph_traits.hpp>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
namespace modelica
{
	template<typename Graph, typename Iterator, typename Value>
	class GraphIterator
	{
		public:
		using iterator_category = std::random_access_iterator_tag;
		using value_type = Value;
		using difference_type = std::ptrdiff_t;
		using pointer = Value*;
		using reference = Value&;

		GraphIterator(Graph& graph, Iterator iter): graph(graph), iter(iter) {}

		[[nodiscard]] bool operator==(const GraphIterator& other) const
		{
			return iter == other.iter;
		}
		[[nodiscard]] bool operator!=(const GraphIterator& other) const
		{
			return iter != other.iter;
		}
		[[nodiscard]] reference operator*() const { return graph[*iter]; }
		[[nodiscard]] reference operator*() { return graph[*iter]; }
		[[nodiscard]] pointer operator->() { return &graph[*iter]; }
		[[nodiscard]] pointer operator->() const { return &graph[*iter]; }
		const GraphIterator operator++(int)	 // NOLINT
		{
			auto copy = *this;
			++(*this);
			return copy;
		}
		GraphIterator& operator++()
		{
			iter++;
			return *this;
		}
		const GraphIterator operator--(int)	 // NOLINT
		{
			auto copy = *this;
			--(*this);
			return copy;
		}
		GraphIterator& operator--()
		{
			iter--;
			return *this;
		}

		private:
		Graph& graph;
		Iterator iter;
	};

	template<typename VertexIndex>
	class Scc
	{
		public:
		using Vector = llvm::SmallVector<VertexIndex, 3>;
		using Iter = typename Vector::iterator;
		using ConstIter = typename Vector::const_iterator;

		Scc(Vector indexes): indexes(std::move(indexes)) {}
		Scc() = default;

		template<typename Graph>
		[[nodiscard]] auto range(const Graph& toIterateOver) const
		{
			using VertexDesc =
					typename std::remove_reference<decltype(toIterateOver[0])>::type;
			using Iterator = GraphIterator<const Graph, ConstIter, const VertexDesc>;

			auto begin = Iterator(toIterateOver, indexes.begin());
			auto end = Iterator(toIterateOver, indexes.end());
			return llvm::make_range<Iterator>(begin, end);
		}

		void push_back(VertexIndex index) { indexes.push_back(index); }

		[[nodiscard]] size_t size() const { return indexes.size(); }

		[[nodiscard]] VertexIndex operator[](size_t index) const
		{
			return indexes[index];
		}

		template<typename Graph>
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

	template<typename VertexIndex>
	class SccLookup
	{
		public:
		using SCC = Scc<VertexIndex>;
		using Vector = llvm::SmallVector<SCC, 3>;
		using InputVector = llvm::SmallVector<VertexIndex, 3>;

		SccLookup(InputVector vector, size_t componentsCount)
				: components(componentsCount), directLookup(std::move(vector))
		{
			for (auto i : irange(directLookup.size()))
			{
				const auto scc = directLookup[i];
				components[scc].push_back(i);
			}
		}

		[[nodiscard]] size_t count() const { return components.size(); }

		[[nodiscard]] auto begin() const { return components.begin(); }
		[[nodiscard]] auto end() const { return components.end(); }
		[[nodiscard]] auto begin() { return components.begin(); }
		[[nodiscard]] auto end() { return components.end(); }
		[[nodiscard]] const SCC& sccOf(size_t vertexIndex) const
		{
			return components[directLookup[vertexIndex]];
		}
		[[nodiscard]] const SCC& operator[](size_t index) const
		{
			return components[index];
		}

		private:
		Vector components;
		InputVector directLookup;
	};
}	 // namespace modelica
