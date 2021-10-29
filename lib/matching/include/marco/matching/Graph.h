#ifndef MARCO_MATCHING_MATCHINGGRAPH_H
#define MARCO_MATCHING_MATCHINGGRAPH_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/ADT/SmallVector.h>
#include <type_traits>
#include <variant>

#include "AccessFunction.h"
#include "IncidenceMatrix.h"
#include "LocalMatchingSolutions.h"
#include "MCIM.h"
#include "MCIS.h"
#include "Range.h"

namespace marco::matching
{
	namespace detail
	{
		template<typename V>
		struct VertexDescriptor
		{
			V* data;
		};

		template<typename V, typename E>
		class EdgeDescriptor
		{
			E* data;
			VertexDescriptor<V> first;
			VertexDescriptor<V> second;
		};

		template<class VariableProperty>
		class VariableVertex
		{
			public:
			using Id = typename VariableProperty::Id;

			VariableVertex(VariableProperty property)
					: property(std::move(property)),
            match(MultidimensionalRange(Range(0, 1)), getRanges())
			{
				assert(getRank() > 0 && "Scalar variables are not supported");
			}

			VariableProperty& getProperty()
			{
				return property;
			}

			const VariableProperty& getProperty() const
			{
				return property;
			}

			VariableVertex::Id getId() const
			{
				return property.getId();
			}

			unsigned int getRank() const
			{
				return property.getRank();
			}

			long getDimensionSize(size_t index) const
			{
				assert(index < getRank());
				return property.getDimensionSize(index);
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

      IncidenceMatrix& getMatchMatrix()
      {
        return match;
      }

      const IncidenceMatrix& getMatchMatrix() const
      {
        return match;
      }

      bool allComponentsMatched() const
      {
        auto& variableRanges = match.getVariableRanges();

        llvm::SmallVector<long> indexes(1 + variableRanges.rank(), 0);

        for (const auto& variableIndexes : match.getVariableRanges())
        {
          for (const auto& index : llvm::enumerate(variableIndexes))
            indexes[1 + index.index()] = index.value();

          if (!match.get(indexes))
            return false;
        }

        return true;
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
			VariableProperty property;
      IncidenceMatrix match;
      bool visible;
		};
	}

	template<typename VariableProperty>
	class Access
	{
		public:
		Access(VariableProperty variable, AccessFunction accessFunction)
				: variable(std::move(variable)),
					accessFunction(std::move(accessFunction))
		{
		}

		template<typename... T>
		Access(VariableProperty variable, T&&... accesses)
				: variable(std::move(variable)),
					accessFunction(llvm::ArrayRef<DimensionAccess>({ std::forward<T>(accesses)... }))
		{
		}

		VariableProperty getVariable() const
		{
			return variable;
		}

		AccessFunction getAccessFunction() const
		{
			return accessFunction;
		}

		private:
		VariableProperty variable;
		AccessFunction accessFunction;
	};

	namespace detail
	{
		template<class EquationProperty, class VariableProperty>
		class EquationVertex
		{
			public:
			using Id = typename EquationProperty::Id;

			EquationVertex(EquationProperty property)
					: property(std::move(property)),
            match(getIterationRanges(), MultidimensionalRange(Range(0, 1)))
			{
			}

			EquationProperty& getProperty()
			{
				return property;
			}

			const EquationProperty& getProperty() const
			{
				return property;
			}

			Id getId() const
			{
				return property.getId();
			}

			unsigned int getNumOfIterationVars() const
			{
				return property.getNumOfIterationVars();
			}

			Range getIterationRange(size_t index) const
			{
				assert(index < getNumOfIterationVars());
				return property.getIterationRange(index);
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

			void getVariableAccesses(llvm::SmallVectorImpl<Access<VariableProperty>>& accesses) const
			{
				property.getVariableAccesses(accesses);
			}

      IncidenceMatrix& getMatchMatrix()
      {
        return match;
      }

      const IncidenceMatrix& getMatchMatrix() const
      {
        return match;
      }

      bool allComponentsMatched() const
      {
        auto& equationRanges = match.getEquationRanges();

        llvm::SmallVector<long> indexes(equationRanges.rank() + 1, 0);

        for (const auto& equationIndexes : equationRanges)
        {
          for (const auto& index : llvm::enumerate(equationIndexes))
            indexes[index.index()] = index.value();

          if (!match.get(indexes))
            return false;
        }

        return true;
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
			EquationProperty property;
      IncidenceMatrix match;
      bool visible;
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

			void addMatch(IncidenceMatrix match)
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

    template<typename ValueType, typename EdgeIterator>
    class VisibleEdgeIterator
    {
      public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = ValueType;
      using difference_type = std::ptrdiff_t;
      using pointer = ValueType*;
      using reference = ValueType&;

      VisibleEdgeIterator(EdgeIterator current, EdgeIterator end) : current(std::move(current)), end(std::move(end))
      {
        fetchNext();
      }

      operator bool() const
      {
        return current != end;
      }

      bool operator==(const VisibleEdgeIterator& it) const
      {
        return current == it.current && end == it.end;
      }

      bool operator!=(const VisibleEdgeIterator& it) const
      {
        return current != it.current || end != it.end;
      }

      VisibleEdgeIterator& operator++()
      {
        fetchNext();
        return *this;
      }

      VisibleEdgeIterator operator++(int)
      {
        auto temp = *this;
        fetchNext();
        return temp;
      }

      value_type operator*()
      {
        return *current;
      }

      private:
      void fetchNext()
      {
        while (current != end && !current->isVisible())
          ++current;
      }

      EdgeIterator current;
      EdgeIterator end;
    };
	}

	template<
			class VariableProperty,
			class EquationProperty>
	class MatchingGraph
	{
		public:
		using Variable = detail::VariableVertex<VariableProperty>;
		using Equation = detail::EquationVertex<EquationProperty, VariableProperty>;
		using Vertex = std::variant<Variable, Equation>;
		using Edge = detail::Edge;

		private:
		using Graph = boost::adjacency_list<
				boost::setS,        // OutEdgeList = set (no multigraph)
				boost::listS,       // VertexList  = list (efficient node removal)
				boost::undirectedS, // Graph is undirected
				Vertex,
				Edge>;

		public:
		using VertexDescriptor = typename boost::graph_traits<Graph>::vertex_descriptor;
		using EdgeDescriptor = typename boost::graph_traits<Graph>::edge_descriptor;

		private:
		using VertexIterator = typename boost::graph_traits<Graph>::vertex_iterator;
		using EdgeIterator = typename boost::graph_traits<Graph>::edge_iterator;

		public:
		bool hasVariable(typename Variable::Id id) const
		{
			return hasVertex<Variable>(id);
		}

    bool isVariable(VertexDescriptor vertex) const
    {
      return std::holds_alternative<Variable>(graph[vertex]);
    }

		VertexDescriptor getVariableVertex(typename Variable::Id id) const
		{
			auto search = findVertex<Variable>(id);
			assert(search.first && "Variable not found");
			return *search.second;
		}

		VariableProperty& getVariable(typename Variable::Id id)
		{
			auto vertex = getVariableVertex(id);
			return std::get<Variable>(graph[vertex]).getProperty();
		}

		const VariableProperty& getVariable(typename Variable::Id id) const
		{
			auto vertex = getVariableVertex(id);
			return std::get<Variable>(graph[vertex]).getProperty();
		}

    Variable& getVariable(VertexDescriptor vertex)
    {
      assert(isVariable(vertex));
      return std::get<Variable>(graph[vertex]);
    }

    const Variable& getVariable(VertexDescriptor vertex) const
    {
      assert(isVariable(vertex));
      return std::get<Variable>(graph[vertex]);
    }

		void addVariable(VariableProperty property)
		{
			Variable variable(std::move(property));
			assert(!hasVariable(variable.getId()) && "Already existing variable");
			boost::add_vertex(std::move(variable), graph);
		}

		bool hasEquation(typename Equation::Id id) const
		{
			return hasVertex<Equation>(id);
		}

    bool isEquation(VertexDescriptor vertex) const
    {
      return std::holds_alternative<Equation>(graph[vertex]);
    }

    VertexDescriptor getEquationVertex(typename Equation::Id id) const
		{
			auto search = findVertex<Equation>(id);
			assert(search.first && "Equation not found");
			return *search.second;
		}

		EquationProperty& getEquation(typename Equation::Id id)
		{
			auto vertex = getEquationVertex(id);
			return std::get<Equation>(graph[vertex]).getProperty();
		}

		const EquationProperty& getEquation(typename Equation::Id id) const
		{
			auto vertex = getEquationVertex(id);
			return std::get<Equation>(graph[vertex]).getProperty();
		}

    Equation& getEquation(VertexDescriptor vertex)
    {
      assert(isEquation(vertex));
      return std::get<Equation>(graph[vertex]);
    }

    const Equation& getEquation(VertexDescriptor vertex) const
    {
      assert(isEquation(vertex));
      return std::get<Equation>(graph[vertex]);
    }

		void addEquation(EquationProperty property)
		{
			Equation equation(std::move(property));
			assert(!hasEquation(equation.getId()) && "Already existing equation");
			auto equationVertex = boost::add_vertex(equation, graph);

			llvm::SmallVector<Access<VariableProperty>, 3> accesses;
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

		size_t getNumberOfScalarVariables() const
		{
			size_t result = 0;

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

		size_t getNumberOfScalarEquations() const
		{
			size_t result = 0;

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

		auto getVertices()
		{
			return boost::make_iterator_range(boost::vertices(graph));
		}

		bool hasEdge(typename EquationProperty::Id equationId, typename VariableProperty::Id variableId) const
		{
			return findEdge<Equation, Variable>(equationId, variableId).first;
		}

		EdgeDescriptor getFirstOutEdge(VertexDescriptor vertex) const
		{
			auto edges = boost::out_edges(vertex, graph);
			assert(edges.first != edges.second && "Vertex doesn't belong to any edge");
			return *edges.first;
		}

		std::pair<VertexDescriptor, VertexDescriptor> getEdgeVertices(EdgeDescriptor edge)
		{
			return std::make_pair(boost::source(edge, graph), boost::target(edge, graph));
		}

    bool simplify()
    {
      // Vertices that are candidate for the first simplification phase.
      // They are the ones having only one incident edge.
      std::list<VertexDescriptor> candidates;

      for (auto& vertex : getVertices())
      {
        auto incidentEdges = getVertexVisibilityDegree(vertex);

        if (incidentEdges == 0)
          return false;

        if (incidentEdges == 1)
          candidates.push_back(vertex);
      }

      while (!candidates.empty())
      {
        VertexDescriptor v1 = candidates.front();
        candidates.pop_front();

        auto edgeDescriptor = getFirstOutEdge(v1);
        Edge& edge = getEdge(edgeDescriptor);

        auto vertices = getEdgeVertices(edgeDescriptor);
        VertexDescriptor v2 = vertices.first == v1 ? vertices.second : vertices.first;

        const auto& u = edge.getIncidenceMatrix();

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

          Variable& variable = isVariable(v1) ? getVariable(v1) : getVariable(v2);
          Equation& equation = isEquation(v1) ? getEquation(v1) : getEquation(v2);

          variable.getMatchMatrix() += matchOptions[0].flattenEquations();
          equation.getMatchMatrix() += matchOptions[0].flattenVariables();

          auto allComponentsMatchedVisitor = [](const auto& vertex) -> bool {
            return vertex.allComponentsMatched();
          };

          if (!std::visit(allComponentsMatchedVisitor, getVertex(v1)))
            return false;

          bool shouldRemoveOppositeNode = std::visit(allComponentsMatchedVisitor, getVertex(v2));

          // Hide the edge
          edge.setVisibility(false);

          // Hide the v1 vertex
          std::visit([](auto& obj) {
            obj.setVisibility(false);
          }, getVertex(v1));

          if (shouldRemoveOppositeNode)
          {
            for (auto v2Edge : getEdges(v2))
            {
              getEdge(v2Edge).setVisibility(false);

              auto v2EdgeVertices = getEdgeVertices(edgeDescriptor);
              VertexDescriptor other = v2EdgeVertices.first == v2 ? v2EdgeVertices.second : v2EdgeVertices.first;

              auto isOtherVertexVisibleFn = [](const auto& obj) -> bool {
                return obj.isVisible();
              };

              if (!std::visit(isOtherVertexVisibleFn, getVertex(other)))
                continue;

              switch (getVertexVisibilityDegree(other))
              {
                case 0:
                  return false;

                case 1:
                  candidates.push_back(other);

                default:
                  break;
              }
            }

            // Hide the v2 vertex and remove it from the candidates
            std::visit([](auto& obj) {
                obj.setVisibility(false);
            }, getVertex(v2));

            candidates.remove_if([&](auto v) {
              return v == v2;
            });
          }
          else
          {
            switch (getVertexVisibilityDegree(v2))
            {
              case 0:
                return false;

              case 1:
                candidates.push_back(v2);
                break;

              default:
                break;
            }
          }
        }
      }

      return true;
    }

		private:
		Vertex& getVertex(VertexDescriptor vertex)
    {
      return graph[vertex];
    }

    const Vertex& getVertex(VertexDescriptor vertex) const
    {
      return graph[vertex];
    }

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

    size_t getVertexVisibilityDegree(VertexDescriptor vertex) const
    {
      size_t result = 0;

      for (auto edge : getEdges(vertex))
        if (getEdge(edge).isVisible())
          ++result;

      return result;
    }

    Edge& getEdge(EdgeDescriptor edge)
    {
      return graph[edge];
    }

    const Edge& getEdge(EdgeDescriptor edge) const
    {
      return graph[edge];
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

    auto getEdges(VertexDescriptor vertex) const
    {
      auto iterators = boost::out_edges(vertex, graph);
      return llvm::iterator_range(iterators.first, iterators.second);
    }

    auto getVisibleEdges(VertexDescriptor vertex) const
    {
      auto iterators = boost::out_edges(vertex, graph);

      detail::VisibleEdgeIterator<Edge, EdgeIterator> visibleBegin(iterators.first, iterators.second);
      detail::VisibleEdgeIterator<Edge, EdgeIterator> visibleEnd(iterators.second, iterators.second);

      return llvm::iterator_range(visibleBegin, visibleEnd);
    }

		Graph graph;
	};
}

#endif	// MARCO_MATCHING_MATCHINGGRAPH_H
