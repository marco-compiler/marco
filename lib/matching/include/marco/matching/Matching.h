#ifndef MARCO_MATCHING_MATCHING_H
#define MARCO_MATCHING_MATCHING_H

#include "AccessFunction.h"
#include "Graph.h"
#include "IncidenceMatrix.h"
#include "Range.h"

namespace marco::matching
{
	namespace detail
	{
		class LocalMatchingSolutions;

		template<typename Container, typename ValueType>
		class LocalMatchingSolutionsIterator
		{
			public:
			using iterator_category = std::forward_iterator_tag;
			using value_type = ValueType;
			using difference_type = std::ptrdiff_t;
			using pointer = ValueType*;
			using reference = ValueType&;

			LocalMatchingSolutionsIterator(
					Container& container,
					size_t index)
					: container(&container),
						index(std::move(index))
			{
			}

			operator bool() const
			{
				return index != container->size();
			}

			bool operator==(const LocalMatchingSolutionsIterator& it) const
			{
				return index == it.index && container == it.container;
			}

			bool operator!=(const LocalMatchingSolutionsIterator& it) const
			{
				return index != it.index || container != it.container;
			}

			LocalMatchingSolutionsIterator& operator++()
			{
				index = std::min(index + 1, container->size());
				return *this;
			}

			LocalMatchingSolutionsIterator operator++(int)
			{
				auto temp = *this;
				index = std::min(index + 1, container->size());
				return temp;
			}

			value_type operator*()
			{
				return (*container)[index];
			}

			private:
			Container* container;
			size_t index;
		};

		class LocalMatchingSolutions
		{
			public:
			using iterator = LocalMatchingSolutionsIterator<
					LocalMatchingSolutions, IncidenceMatrix>;

			using const_iterator = LocalMatchingSolutionsIterator<
					const LocalMatchingSolutions, const IncidenceMatrix>;

			LocalMatchingSolutions(
					llvm::ArrayRef<AccessFunction> accessFunctions,
					MultidimensionalRange equationRanges,
					MultidimensionalRange variableRanges);

			IncidenceMatrix& operator[](size_t index);
			const IncidenceMatrix& operator[](size_t index) const;

			size_t size() const;

			iterator begin();
			const_iterator begin() const;

			iterator end();
			const_iterator end() const;

			private:
			void compute();

			void getInductionVariablesUsage(
					llvm::SmallVectorImpl<size_t>& usages,
					const AccessFunction& accessFunction) const;

			llvm::ArrayRef<AccessFunction> accessFunctions;
			MultidimensionalRange equationRanges;
			MultidimensionalRange variableRanges;
			size_t solutionsCount;
			llvm::SmallVector<IncidenceMatrix, 3> matrices;
		};

		detail::LocalMatchingSolutions solveLocalMatchingProblem(
				const IncidenceMatrix& u,
				llvm::ArrayRef<AccessFunction> accessFunctions);
	}

	template<
			class VariableDescriptor,
			class EquationDescriptor>
	bool simplify(MatchingGraph<VariableDescriptor, EquationDescriptor>& graph)
	{
		using Graph = MatchingGraph<VariableDescriptor, EquationDescriptor>;
		using VertexDescriptor = typename Graph::VertexDescriptor;
		using EdgeDescriptor = typename Graph::EdgeDescriptor;
		using Edge = typename Graph::Edge;

		// Vertices that are candidate for the first simplification phase.
		// They are the ones having only one incident edge.
		std::list<VertexDescriptor> candidates;

		for (auto& vertex : graph.getVertices())
		{
			auto incidentEdges = graph.getVertexVisibilityDegree(vertex);

			if (incidentEdges == 0)
				return false;

			if (incidentEdges == 1)
				candidates.push_back(vertex);
		}

		while (!candidates.empty())
		{
			VertexDescriptor v1 = candidates.front();
			candidates.pop_front();

			auto edgeDescriptor = graph.getFirstOutEdge(v1);
			auto vertices = graph.getEdgeVertices(edgeDescriptor);
			VertexDescriptor v2 = vertices.first == v1 ? vertices.second : vertices.first;
			bool shouldRemoveOppositeNode = false;

			Edge& edge = graph[edgeDescriptor];
			const auto& u = edge.getIncidenceMatrix();
			std::cout << "u\n" << u << "\n\n";

			auto matchOptions = detail::solveLocalMatchingProblem(u, edge.getAccessFunctions());

			for (const auto& m : matchOptions)
				std::cout << "m\n" << m << "\n\n";

			// The simplification steps is executed only in case of a single
			// matching option. In case of multiple ones, in fact, the choice
			// would be arbitrary and may affect the feasibility of the
			// array-aware matching problem.

			if (matchOptions.size() == 1)
			{
				//graph[edge].addMatch(matchOptions.front());

				if (shouldRemoveOppositeNode)
				{

				}
				else
				{

				}
			}
		}

		return true;
	}
}

#endif	// MARCO_MATCHING_MATCHING_H
