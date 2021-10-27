#ifndef MARCO_MATCHING_GRAPH_H
#define MARCO_MATCHING_GRAPH_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <type_traits>
#include <variant>

#include "AccessFunction.h"
#include "IncidenceMatrix.h"
#include "MCIM.h"
#include "MCIS.h"
#include "Range.h"

namespace marco::matching
{
	namespace detail
	{
		template<class VariableDescriptor>
		class VariableVertex
		{
			public:
			using Id = typename VariableDescriptor::Id;

			VariableVertex(VariableDescriptor descriptor)
					: descriptor(std::move(descriptor))
			{
				assert(getRank() > 0 && "Scalar variables are not supported");
			}

			VariableVertex::Id getId() const
			{
				return descriptor.getId();
			}

			unsigned int getRank() const
			{
				return descriptor.getRank();
			}

			long getDimensionSize(size_t index) const
			{
				assert(index < getRank());
				return descriptor.getDimensionSize(index);
			}

			MultidimensionalRange getRanges() const
			{
				llvm::SmallVector<Range, 3> ranges;

				for (size_t i = 0; i < getRank(); ++i)
				{
					long size = getDimensionSize(i);
					assert(size > 0);
					ranges.emplace_back(0, size);
				}

				return MultidimensionalRange(ranges);
			}

			unsigned int flatSize() const
			{
				unsigned int result = 1;

				for (unsigned int i = 0, rank = getRank(); i < rank; ++i)
				{
					long size = getDimensionSize(i);
					assert(size > 0);
					result *= size;
				}

				return result;
			}

			private:
			VariableDescriptor descriptor;
		};
	}

	template<typename VariableDescriptor>
	class Access
	{
		public:
		Access(VariableDescriptor variable, AccessFunction accessFunction)
				: variable(std::move(variable)),
					accessFunction(std::move(accessFunction))
		{
		}

		template<typename... T>
		Access(VariableDescriptor variable, T&&... accesses)
				: variable(std::move(variable)),
					accessFunction(llvm::ArrayRef<DimensionAccess>({ std::forward<T>(accesses)... }))
		{
		}

		VariableDescriptor getVariable() const
		{
			return variable;
		}

		AccessFunction getAccessFunction() const
		{
			return accessFunction;
		}

		private:
		VariableDescriptor variable;
		AccessFunction accessFunction;
	};

	namespace detail
	{
		template<class EquationDescriptor, class VariableDescriptor>
		class EquationVertex
		{
			public:
			using Id = typename EquationDescriptor::Id;

			EquationVertex(EquationDescriptor descriptor)
					: descriptor(std::move(descriptor))
			{
			}

			Id getId() const
			{
				return descriptor.getId();
			}

			unsigned int getNumOfIterationVars() const
			{
				return descriptor.getNumOfIterationVars();
			}

			Range getIterationRange(size_t index) const
			{
				assert(index < getNumOfIterationVars());
				return descriptor.getIterationRange(index);
			}

			MultidimensionalRange getIterationRanges() const
			{
				llvm::SmallVector<Range, 3> ranges;

				for (unsigned int i = 0, e = getNumOfIterationVars(); i < e; ++i)
					ranges.push_back(getIterationRange(i));

				return MultidimensionalRange(ranges);
			}

			unsigned int flatSize() const
			{
				return getIterationRanges().flatSize();
			}

			void getVariableAccesses(llvm::SmallVectorImpl<Access<VariableDescriptor>>& accesses) const
			{
				descriptor.getVariableAccesses(accesses);
			}

			private:
			EquationDescriptor descriptor;
		};

		class Edge
		{
			public:
			Edge(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
					: equations(equationRanges.flatSize()),
						variables(variableRanges.flatSize()),
						incidenceMatrix(equationRanges, variableRanges),
						matchMatrix(equationRanges, variableRanges),
						visible(true)
			{
			}

			unsigned int getNumberOfEquations() const
			{
				return equations;
			}

			unsigned int getNumberOfVariables() const
			{
				return variables;
			}

			llvm::ArrayRef<AccessFunction> getAccessFunctions() const
			{
				return accessFunctions;
			}

			void addAccessFunction(AccessFunction accessFunction)
			{
				accessFunctions.push_back(accessFunction);
				incidenceMatrix.apply(accessFunction);
			}

			const IncidenceMatrix& getIncidenceMatrix() const
			{
				return incidenceMatrix;
			}

			void setMatch(IncidenceMatrix match)
			{
				matchMatrix += match;
			}

			bool isVisible() const
			{
				return visible;
			}

			void setVisibility(bool visibility)
			{
				visible = visibility;
			}

			private:
			unsigned int equations;
			unsigned int variables;
			llvm::SmallVector<AccessFunction, 3> accessFunctions;
			IncidenceMatrix incidenceMatrix;
			IncidenceMatrix matchMatrix;
			bool visible;
		};
	}

	template<
			class VariableDescriptor,
			class EquationDescriptor>
	class MatchingGraph
	{
		public:
		using Variable = detail::VariableVertex<VariableDescriptor>;
		using Equation = detail::EquationVertex<EquationDescriptor, VariableDescriptor>;
		using Edge = detail::Edge;

		private:
		using Graph = boost::adjacency_list<
				boost::setS,        // OutEdgeList = set (no multigraph)
				boost::listS,       // VertexList  = list (efficient node removal)
				boost::undirectedS, // Graph is undirected
				std::variant<Variable, Equation>,
				detail::Edge>;

		public:
		using VertexDescriptor = typename boost::graph_traits<Graph>::vertex_descriptor;
		using EdgeDescriptor = typename boost::graph_traits<Graph>::edge_descriptor;

		private:
		using VertexIterator = typename boost::graph_traits<Graph>::vertex_iterator;
		using EdgeIterator = typename boost::graph_traits<Graph>::edge_iterator;

		public:
		Edge& operator[](EdgeDescriptor edge)
		{
			return graph[edge];
		}

		const Edge& operator[](EdgeDescriptor edge) const
		{
			return graph[edge];
		}

		bool hasVariable(typename Variable::Id id) const
		{
			return hasVertex<Variable>(id);
		}

		Variable getVariable(typename Variable::Id id) const
		{
			auto vertex = getVariableVertex(id);
			return std::get<Variable>(graph[vertex]);
		}

		void addVariable(VariableDescriptor descriptor)
		{
			Variable variable(descriptor);
			assert(!hasVariable(variable.getId()) && "Already existing variable");
			boost::add_vertex(std::move(variable), graph);
		}

		bool hasEquation(typename Equation::Id id) const
		{
			return hasVertex<Equation>(id);
		}

		Equation getEquation(typename Equation::Id id) const
		{
			auto vertex = getEquationVertex(id);
			return std::get<Equation>(graph[vertex]);
		}

		void addEquation(EquationDescriptor descriptor)
		{
			Equation equation(descriptor);
			assert(!hasEquation(equation.getId()) && "Already existing equation");
			auto equationVertex = boost::add_vertex(equation, graph);

			llvm::SmallVector<Access<VariableDescriptor>, 3> accesses;
			equation.getVariableAccesses(accesses);

			// The equation may access multiple variables or even multiple indexes
			// of the same variable. Add an edge to the graph for each of those
			// accesses.

			for (const auto& access : accesses)
			{
				auto variableVertex = getVariableVertex(access.getVariable().getId());
				unsigned int numberOfEquations = equation.getIterationRanges().flatSize();
				auto& variable = std::get<Variable>(graph[variableVertex]);
				unsigned int numberOfVariables = variable.flatSize();

				Edge edge(equation.getIterationRanges(), variable.getRanges());
				auto edgeDescriptor = boost::add_edge(equationVertex, variableVertex, edge, graph);

				graph[edgeDescriptor.first].addAccessFunction(access.getAccessFunction());
			}
		}

		unsigned int getNumberOfScalarEquations() const
		{
			unsigned int result = 0;

			for (const auto& v : boost::make_iterator_range(boost::vertices(graph)))
			{
				if (auto& vertex = graph[v]; std::holds_alternative<Equation>(vertex))
				{
					auto& equation = std::get<Equation>(vertex);
					result += equation.flatSize();
				}
			}

			return result;
		}

		unsigned int getNumberOfScalarVariables() const
		{
			unsigned int result = 0;

			for (const auto& v : boost::make_iterator_range(boost::vertices(graph)))
			{
				if (auto& vertex = graph[v]; std::holds_alternative<Variable>(vertex))
				{
					auto& variable = std::get<Variable>(vertex);
					result += variable.flatSize();
				}
			}

			return result;
		}

		auto getVertices()
		{
			return boost::make_iterator_range(boost::vertices(graph));
		}

		unsigned int getVertexVisibilityDegree(const VertexDescriptor& vertex) const
		{
			return boost::out_degree(vertex, graph);
		}

		bool hasEdge(typename EquationDescriptor::Id equationId, typename VariableDescriptor::Id variableId) const
		{
			return findEdge<Equation, Variable>(equationId, variableId).first;
		}

		EdgeDescriptor getFirstOutEdge(const VertexDescriptor& vertex) const
		{
			for (const auto& edge : boost::make_iterator_range(boost::out_edges(vertex, graph)))
				return edge;

			assert(false && "Unreachable");
		}

		std::pair<VertexDescriptor, VertexDescriptor> getEdgeVertices(EdgeDescriptor edge)
		{
			return std::make_pair(boost::source(edge, graph), boost::target(edge, graph));
		}

		/*
		bool simplify()
		{
			// Vertices that are candidate for the first simplification phase.
			// They are the ones having only one incident edge.
			std::list<Vertex> candidates;

			for (Vertex& vertex : boost::make_iterator_range(boost::vertices(graph)))
			{
				auto incidentEdges = getVertexVisibilityDegree(vertex);

				if (incidentEdges == 0)
					return false;

				if (incidentEdges == 1)
					candidates.push_back(vertex);
			}

			while (!candidates.empty())
			{
				Vertex v1 = candidates.front();
				candidates.pop_front();

				Edge edge = getFirstOutEdge(v1);
				Vertex v2 = (boost::source(edge, graph) == v1) ? boost::target(edge, graph) : boost::source(edge, graph);
				bool shouldRemoveOppositeNode = false;

				const auto& u = graph[edge].getIncidenceMatrix();
				std::cout << "u\n" << u << "\n\n";

				auto matchOptions = solveLocalMatchingProblem(edge);

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
		 */

		private:
		template<typename T>
		bool hasVertex(typename T::Id id) const
		{
			return findVertex<T>(id).first;
		}

		template<typename T>
		std::pair<bool, VertexIterator> findVertex(typename T::Id id) const
		{
			VertexIterator begin, end;
			std::tie(begin, end) = boost::vertices(graph);

			auto it = std::find_if(begin, end, [&](const VertexDescriptor& v) {
				const auto& vertex = graph[v];

				if (!std::holds_alternative<T>(vertex))
					return false;

				return std::get<T>(vertex).getId() == id;
			});

			return std::make_pair(it != end, it);
		}

		VertexDescriptor getVariableVertex(typename Variable::Id id) const
		{
			auto search = findVertex<Variable>(id);
			assert(search.first && "Variable not found");
			return *search.second;
		}

		VertexDescriptor getEquationVertex(typename Equation::Id id) const
		{
			auto search = findVertex<Equation>(id);
			assert(search.first && "Equation not found");
			return *search.second;
		}

		template<typename From, typename To>
		std::pair<bool, EdgeIterator> findEdge(typename From::Id from, typename To::Id to) const
		{
			EdgeIterator begin, end;
			std::tie(begin, end) = boost::edges(graph);

			auto it = std::find_if(begin, end, [&](const EdgeDescriptor& e) {
				auto& source = graph[boost::source(e, graph)];
				auto& target = graph[boost::target(e, graph)];

				if (!std::holds_alternative<From>(source) || !std::holds_alternative<To>(target))
					return false;

				return std::get<From>(source).getId() == from && std::get<To>(target).getId() == to;
			});

			return std::make_pair(it != end, it);
		}

		Graph graph;
	};
}

#endif	// MARCO_MATCHING_GRAPH_H
