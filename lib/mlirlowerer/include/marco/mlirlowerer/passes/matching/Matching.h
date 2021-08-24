#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_selectors.hpp>
#include <iterator>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Error.h>
#include <map>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/utils/IndexSet.hpp>
#include <utility>

namespace marco::codegen::model
{
	class Edge;
	class ExpressionPath;

	class MatchingGraph
	{
		template<typename MatchingGraph, typename Iterator, typename EdgeType>
		class MatchingGraphIterator
		{
			public:
			using iterator_category = typename Iterator::iterator_category;
			using value_type = EdgeType&;
			using difference_type = typename Iterator::difference_type;
			using pointer = EdgeType*;
			using reference = EdgeType&;

			MatchingGraphIterator(MatchingGraph& graph, Iterator iter)
					: graph(&graph), iter(iter)
			{
			}

			[[nodiscard]] bool operator==(const MatchingGraphIterator& other) const
			{
				return iter == other.iter;
			}

			[[nodiscard]] bool operator!=(const MatchingGraphIterator& other) const
			{
				return iter != other.iter;
			}

			[[nodiscard]] reference operator*() const { return (*graph)[*iter]; }
			[[nodiscard]] reference operator*() { return (*graph)[*iter]; }
			[[nodiscard]] pointer operator->() { return &(*graph)[*iter]; }
			[[nodiscard]] pointer operator->() const { return &(*graph)[*iter]; }

			const MatchingGraphIterator operator++(int)	 // NOLINT
			{
				auto copy = *this;
				++(*this);
				return copy;
			}

			MatchingGraphIterator& operator++()
			{
				iter++;
				return *this;
			}

			const MatchingGraphIterator operator--(int)	 // NOLINT
			{
				auto copy = *this;
				--(*this);
				return copy;
			}

			MatchingGraphIterator& operator--()
			{
				iter--;
				return *this;
			}

			private:
			MatchingGraph* graph;
			Iterator iter;
		};

		public:
		using GraphImp = boost::adjacency_list<
				boost::vecS,
				boost::vecS,
				boost::undirectedS,
				boost::no_property,
				Edge>;

		using EdgeDesc = boost::graph_traits<GraphImp>::edge_descriptor;

		using VertexDesc = boost::graph_traits<GraphImp>::vertex_descriptor;

		using OutEdgeIter = boost::graph_traits<GraphImp>::out_edge_iterator;

		using EdgeIter = boost::graph_traits<GraphImp>::edge_iterator;

		using ConstEdgeIter = boost::graph_traits<const GraphImp>::edge_iterator;

		using ConstOutEdgeIter = boost::graph_traits<const GraphImp>::out_edge_iterator;

		using EquationLookup = std::map<const Equation, VertexDesc>;
		using VariableLookup = std::map<const Variable, VertexDesc>;

		using out_iterator = class MatchingGraphIterator<MatchingGraph, OutEdgeIter, Edge>;

		using const_out_iterator = class MatchingGraphIterator<const MatchingGraph, ConstOutEdgeIter, const Edge>;

		using edge_iterator = class MatchingGraphIterator<MatchingGraph, EdgeIter, Edge>;

		using const_edge_iterator = class MatchingGraphIterator<const MatchingGraph, ConstEdgeIter, const Edge>;

		MatchingGraph(const Model& model);

		[[nodiscard]] const Model& getModel() const;
		[[nodiscard]] size_t variableCount() const;
		[[nodiscard]] size_t equationCount() const;

		[[nodiscard]] Edge& operator[](EdgeDesc desc);
		[[nodiscard]] const Edge& operator[](EdgeDesc desc) const;

		[[nodiscard]] edge_iterator begin();
		[[nodiscard]] const_edge_iterator begin() const;

		[[nodiscard]] edge_iterator end();
		[[nodiscard]] const_edge_iterator end() const;

		[[nodiscard]] mlir::LogicalResult match(unsigned int maxIterations);

		[[nodiscard]] size_t matchedCount() const;
		[[nodiscard]] size_t edgesCount() const;
		[[nodiscard]] size_t matchedEdgesCount() const;

		[[nodiscard]] llvm::iterator_range<out_iterator> arcsOf(const Equation& equation);
		[[nodiscard]] llvm::iterator_range<const_out_iterator> arcsOf(const Equation& equation) const;

		[[nodiscard]] llvm::iterator_range<out_iterator> arcsOf(const Variable& var);
		[[nodiscard]] llvm::iterator_range<const_out_iterator> arcsOf(const Variable& var) const;

		[[nodiscard]] IndexSet getUnmatchedSet(const Variable& variable) const;
		[[nodiscard]] IndexSet getUnmatchedSet(const Equation& equation) const;

		[[nodiscard]] IndexSet getMatchedSet(const Variable& variable) const;
		[[nodiscard]] IndexSet getMatchedSet(const Equation& eq) const;

		template<typename T>
		[[nodiscard]] size_t outDegree(const T& t) const
		{
			auto r = arcsOf(t);
			return std::distance(r.begin(), r.end());
		}

		private:
		void addEquation(Equation eq);
		void emplaceEdge(Equation eq, ExpressionPath path, size_t index);

		VertexDesc getDesc(const Equation& eq);
		VertexDesc getDesc(const Variable& var);

		GraphImp graph;
		EquationLookup equationLookUp;
		VariableLookup variableLookUp;
		const Model& model;
	};

	mlir::LogicalResult match(Model& model, size_t maxIterations);
}
