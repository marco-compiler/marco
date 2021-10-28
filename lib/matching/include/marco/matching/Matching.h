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

			LocalMatchingSolutionsIterator(Container& container, size_t index)
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

			value_type& operator*()
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

			LocalMatchingSolutions(
					llvm::ArrayRef<AccessFunction> accessFunctions,
					MultidimensionalRange equationRanges,
					MultidimensionalRange variableRanges);

			IncidenceMatrix& operator[](size_t index);

			size_t size() const;

			iterator begin();
			iterator end();

			private:
			void fetchNext();

			void getInductionVariablesUsage(
					llvm::SmallVectorImpl<size_t>& usages,
					const AccessFunction& accessFunction) const;

			llvm::SmallVector<AccessFunction, 3> accessFunctions;
			MultidimensionalRange equationRanges;
			MultidimensionalRange variableRanges;

			// Total number of possible match matrices
			size_t solutionsAmount;

			// List of the computed match matrices
			llvm::SmallVector<IncidenceMatrix, 3> matrices;

			size_t currentAccessFunction = 0;
			size_t groupSize;
			llvm::SmallVector<Range, 3> reorderedRanges;
			std::unique_ptr<MultidimensionalRange> range;
			llvm::SmallVector<size_t, 3> ordering;
			std::unique_ptr<MultidimensionalRange::iterator> rangeIt;
			std::unique_ptr<MultidimensionalRange::iterator> rangeEnd;
		};

		LocalMatchingSolutions solveLocalMatchingProblem(
				const MultidimensionalRange& equationRanges,
				const MultidimensionalRange& variableRanges,
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
			Edge& edge = graph.getEdge(edgeDescriptor);

			auto vertices = graph.getEdgeVertices(edgeDescriptor);
			VertexDescriptor v2 = vertices.first == v1 ? vertices.second : vertices.first;

			bool shouldRemoveOppositeNode = false;

			const auto& u = edge.getIncidenceMatrix();
			std::cout << "u\n" << u << "\n\n";

			auto matchOptions = detail::solveLocalMatchingProblem(
					u.getEquationRanges(),
					u.getVariableRanges(),
					edge.getAccessFunctions());

			// The simplification steps is executed only in case of a single
			// matching option. In case of multiple ones, in fact, the choice
			// would be arbitrary and may affect the feasibility of the
			// array-aware matching problem.

			if (matchOptions.size() == 1)
			{
				edge.addMatch(matchOptions[0]);

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
