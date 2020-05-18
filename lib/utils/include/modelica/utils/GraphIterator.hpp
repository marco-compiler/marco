#pragma once
#include <iterator>

/**
 * Iterates over a range, either of vertex descritor or edge descriptor and
 * returns the property rather and the iterator
 */
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
