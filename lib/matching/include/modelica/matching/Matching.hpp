#pragma once
#include <iterator>
#include <map>
#include <utility>

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/matching/Edge.hpp"
#include "modelica/model/ModEquation.hpp"
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
			[[nodiscard]] reference operator*() const
			{
				return graph->edgeAt(iter->second);
			}
			[[nodiscard]] reference operator*()
			{
				return graph->edgeAt(iter->second);
			}
			[[nodiscard]] pointer operator->()
			{
				return &(graph->edgeAt(iter->second));
			}
			[[nodiscard]] pointer operator->() const
			{
				return &(graph->edgeAt(iter->second));
			}
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
		using EquationLookup = std::multimap<const ModEquation*, size_t>;
		using VariableLookup = std::multimap<const ModVariable*, size_t>;

		using eq_iterator = class MatchingGraphIterator<
				MatchingGraph,
				EquationLookup::iterator,
				Edge>;
		using const_eq_iterator = class MatchingGraphIterator<
				const MatchingGraph,
				EquationLookup::const_iterator,
				const Edge>;
		using var_iterator = class MatchingGraphIterator<
				MatchingGraph,
				VariableLookup::iterator,
				Edge>;
		using const_var_iterator = class MatchingGraphIterator<
				const MatchingGraph,
				VariableLookup::const_iterator,
				const Edge>;

		[[nodiscard]] llvm::iterator_range<eq_iterator> arcsOf(
				const ModEquation& equation)
		{
			const ModEquation* eq = &equation;
			auto [begin, end] = equationLookUp.equal_range(eq);
			return llvm::make_range(
					eq_iterator(*this, begin), eq_iterator(*this, end));
		}

		[[nodiscard]] llvm::iterator_range<const_eq_iterator> arcsOf(
				const ModEquation& equation) const
		{
			const ModEquation* eq = &equation;
			auto [begin, end] = equationLookUp.equal_range(eq);
			return llvm::make_range(
					const_eq_iterator(*this, begin), const_eq_iterator(*this, end));
		}

		[[nodiscard]] llvm::iterator_range<var_iterator> arcsOf(
				const ModVariable& var)
		{
			auto [begin, end] = variableLookUp.equal_range(&var);
			return llvm::make_range(
					var_iterator(*this, begin), var_iterator(*this, end));
		}

		[[nodiscard]] llvm::iterator_range<const_var_iterator> arcsOf(
				const ModVariable& var) const
		{
			auto [begin, end] = variableLookUp.equal_range(&var);
			return llvm::make_range(
					const_var_iterator(*this, begin), const_var_iterator(*this, end));
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
			auto set = equation.toIndexSet();
			set.remove(getMatchedSet(equation));
			return set;
		}

		[[nodiscard]] IndexSet getMatchedSet(const ModVariable& variable) const
		{
			IndexSet matched;

			for (const Edge& edge : arcsOf(variable))
				matched.unite(edge.getSet());

			return matched;
		}

		[[nodiscard]] IndexSet getMatchedSet(const ModEquation& eq) const
		{
			IndexSet matched;
			for (const Edge& edge : arcsOf(eq))
				matched.unite(edge.getSet());

			return matched;
		}

		[[nodiscard]] const Edge& edgeAt(size_t index) const
		{
			return edges[index];
		}
		[[nodiscard]] Edge& edgeAt(size_t index) { return edges[index]; }

		[[nodiscard]] auto begin() const { return edges.begin(); }
		[[nodiscard]] auto begin() { return edges.begin(); }
		[[nodiscard]] auto end() { return edges.end(); }
		[[nodiscard]] auto end() const { return edges.end(); }
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
		[[nodiscard]] size_t edgesCount() const { return edges.size(); }
		void dumpGraph(llvm::raw_ostream& OS) const;
		void match(int maxIterations);

		[[nodiscard]] size_t matchedEdgesCount() const;
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		private:
		void addEquation(const ModEquation& eq);

		llvm::SmallVector<Edge, 0> edges;
		EquationLookup equationLookUp;
		VariableLookup variableLookUp;
		const Model& model;
	};

}	 // namespace modelica
