#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_selectors.hpp>
#include <iterator>
#include <map>
#include <utility>

#include "boost/graph/adjacency_list.hpp"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/Edge.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExpPath.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
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

		using ConstOutEdgeIter =
				boost::graph_traits<const GraphImp>::out_edge_iterator;

		using EquationLookup = std::map<const ModEquation*, VertexDesc>;
		using VariableLookup = std::map<const ModVariable*, VertexDesc>;

		using out_iterator =
				class MatchingGraphIterator<MatchingGraph, OutEdgeIter, Edge>;
		using const_out_iterator = class MatchingGraphIterator<
				const MatchingGraph,
				ConstOutEdgeIter,
				const Edge>;

		using edge_iterator =
				class MatchingGraphIterator<MatchingGraph, EdgeIter, Edge>;
		using const_edge_iterator = class MatchingGraphIterator<
				const MatchingGraph,
				ConstEdgeIter,
				const Edge>;

		[[nodiscard]] llvm::iterator_range<out_iterator> arcsOf(
				const ModEquation& equation)
		{
			const auto iter = equationLookUp.find(&equation);
			assert(equationLookUp.end() != iter);
			auto [begin, end] = boost::out_edges(iter->second, graph);
			return llvm::make_range(
					out_iterator(*this, begin), out_iterator(*this, end));
		}

		[[nodiscard]] llvm::iterator_range<const_out_iterator> arcsOf(
				const ModEquation& equation) const
		{
			const auto iter = equationLookUp.find(&equation);
			assert(equationLookUp.end() != iter);
			auto [begin, end] = boost::out_edges(iter->second, graph);
			return llvm::make_range(
					const_out_iterator(*this, begin), const_out_iterator(*this, end));
		}

		[[nodiscard]] llvm::iterator_range<out_iterator> arcsOf(
				const ModVariable& var)
		{
			const auto iter = variableLookUp.find(&var);
			assert(variableLookUp.end() != iter);
			auto [begin, end] = boost::out_edges(iter->second, graph);
			return llvm::make_range(
					out_iterator(*this, begin), out_iterator(*this, end));
		}

		[[nodiscard]] llvm::iterator_range<const_out_iterator> arcsOf(
				const ModVariable& var) const
		{
			const auto iter = variableLookUp.find(&var);
			assert(variableLookUp.end() != iter);
			auto [begin, end] = boost::out_edges(iter->second, graph);
			return llvm::make_range(
					const_out_iterator(*this, begin), const_out_iterator(*this, end));
		}

		MatchingGraph(const Model& model): model(model)
		{
			for (const auto& eq : model)
				addEquation(eq);
		}

		[[nodiscard]] IndexSet getUnmatchedSet(const ModVariable& variable) const
		{
			auto set = variable.toIndexSet();
			set.remove(getMatchedSet(variable));
			return set;
		}

		[[nodiscard]] IndexSet getUnmatchedSet(const ModEquation& equation) const
		{
			IndexSet set(equation.getInductions());
			set.remove(getMatchedSet(equation));
			return set;
		}

		/* restituisce IndexSet relativo agli indici delle equazioni MATCHATE
		 * (forse) */
		[[nodiscard]] IndexSet getMatchedSet(const ModVariable& variable) const
		{
			IndexSet matched;

			/* arcsOf(variables) restituisce SOLO GLI ARCHI MATCHATI */
			for (const Edge& edge : arcsOf(variable))
				matched.unite(edge.getVectorAccess().map(edge.getSet()));

			return matched;
		}

		[[nodiscard]] IndexSet getMatchedSet(const ModEquation& eq) const
		{
			IndexSet matched;
			for (const Edge& edge : arcsOf(eq))
				matched.unite(edge.getSet());

			return matched;
		}

		template<typename T>
		[[nodiscard]] size_t outDegree(const T& t) const
		{
			auto r = arcsOf(t);
			return std::distance(r.begin(), r.end());
		}

		[[nodiscard]] const_edge_iterator begin() const
		{
			return const_edge_iterator(*this, boost::edges(graph).first);
		}
		[[nodiscard]] edge_iterator begin()
		{
			return edge_iterator(*this, boost::edges(graph).first);
		}

		[[nodiscard]] edge_iterator end()
		{
			return edge_iterator(*this, boost::edges(graph).second);
		}
		[[nodiscard]] const_edge_iterator end() const
		{
			return const_edge_iterator(*this, boost::edges(graph).second);
		}

		[[nodiscard]] const Edge& operator[](EdgeDesc desc) const
		{
			return graph[desc];
		}
		[[nodiscard]] Edge& operator[](EdgeDesc desc) { return graph[desc]; }
		[[nodiscard]] const Model& getModel() const { return model; }
		[[nodiscard]] size_t variableCount() const
		{
			return model.getVars().size();
		}
		[[nodiscard]] size_t equationCount() const
		{
			return model.getEquations().size();
		}
		[[nodiscard]] size_t matchedCount() const
		{
			size_t count = 0;
			for (const auto& edge : *this)
				count += edge.getSet().size();

			return count;
		}
		[[nodiscard]] size_t edgesCount() const
		{
			auto [b, e] = boost::edges(graph);
			return std::distance(b, e);
		}
		[[nodiscard]] size_t indexOfEquation(const ModEquation& eq) const;
		void dumpGraph(
				llvm::raw_ostream& OS,
				bool displayEmptyEdges = true,
				bool displayMappings = true,
				bool displayOnlyMatchedCount = true,
				bool closeGraph = true) const;
		void match(int maxIterations);

		[[nodiscard]] size_t matchedEdgesCount() const;
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		private:
		void addEquation(const ModEquation& eq);
		void emplaceEdge(const ModEquation& eq, ModExpPath path, size_t index);
		VertexDesc getDesc(const ModEquation& eq)
		{
			if (equationLookUp.find(&eq) != equationLookUp.end())
				return equationLookUp[&eq];
			auto dec = boost::add_vertex(graph);
			equationLookUp[&eq] = dec;
			return dec;
		}

		VertexDesc getDesc(const ModVariable& var)
		{
			if (variableLookUp.find(&var) != variableLookUp.end())
				return variableLookUp[&var];
			auto dec = boost::add_vertex(graph);
			variableLookUp[&var] = dec;
			return dec;
		}

		GraphImp graph;
		EquationLookup equationLookUp;
		VariableLookup variableLookUp;
		const Model& model;
	};

	llvm::Expected<Model> match(Model entryModel, size_t maxIterations);

}	 // namespace modelica
