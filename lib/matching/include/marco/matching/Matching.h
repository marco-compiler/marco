#ifndef MARCO_MATCHING_MATCHING_H
#define MARCO_MATCHING_MATCHING_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <memory>
#include <type_traits>
#include <variant>

#include "AccessFunction.h"
#include "Graph.h"
#include "IncidenceMatrix.h"
#include "LocalMatchingSolutions.h"
#include "Range.h"

namespace marco::matching
{
  namespace detail
  {
    /**
     * Represent a generic vectorized entity whose scalar elements
     * can be matched with the scalar elements of other arrays.
     * The relationship is tracked by means of an incidence matrix.
     */
    class Matchable
    {
      public:
      Matchable(IncidenceMatrix initialMatch);

      const IncidenceMatrix& getMatched() const;
      IncidenceMatrix getUnmatched() const;

      /**
       * Check whether all the scalar elements of this array
       * have been matched.
       *
       * @return true if all the elements are matched
       */
      bool allComponentsMatched() const;

      void addMatch(IncidenceMatrix newMatch);
      void removeMatch(IncidenceMatrix removedMatch);

      private:
      IncidenceMatrix match;
    };

    /**
     * Graph node representing a variable.
     *
     * Requires the underlying variable property to define the Id type and
     * implement the following methods:
     *  - Id getId() const : get the ID of the variable
     *  - size_t getRank() : get the number of dimensions
     *  - long getDimensionSize(size_t index) : get the size of a dimension
     */
    template<class VariableProperty>
    class VariableVertex : public Matchable
    {
      public:
      using Id = typename VariableProperty::Id;

      VariableVertex(VariableProperty property)
              : Matchable(IncidenceMatrix::row(getRanges(property))),
                property(property),
                visible(true)
      {
        assert(getRank() > 0 && "Scalar variables are not supported");
      }

      void dump() const
      {
        dump(llvm::outs());
      }

      void dump(llvm::raw_ostream& os) const
      {
        os << "Variable\n";
        os << " |-- ID: " << getId() << "\n";
        os << " |-- Rank: " << getRank() << "\n";
        os << " |-- Dimensions: ";

        for (size_t i = 0; i < getRank(); ++i)
          os << "[" << getDimensionSize(i) << "]";

        os << "\n";
        os << " |-- Match matrix:\n";
        os << getMatched();
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

      // TODO: change unsigned int to size_t
      unsigned int getRank() const
      {
        return getRank(property);
      }

      long getDimensionSize(size_t index) const
      {
        return getDimensionSize(property, index);
      }

      MultidimensionalRange getRanges() const
      {
        return getRanges(property);
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

      bool isVisible() const
      {
        return visible;
      }

      void setVisibility(bool visibility)
      {
        visible = visibility;
      }

      private:
      static unsigned int getRank(const VariableProperty& p)
      {
        return p.getRank();
      }

      static long getDimensionSize(const VariableProperty& p, size_t index)
      {
        assert(index < getRank(p));
        return p.getDimensionSize(index);
      }

      static MultidimensionalRange getRanges(const VariableProperty& p)
      {
        llvm::SmallVector<Range, 3> ranges;

        for (size_t i = 0, e = getRank(p); i < e; ++i)
        {
          long size = getDimensionSize(p, i);
          assert(size > 0);
          ranges.emplace_back(0, size);
        }

        return MultidimensionalRange(ranges);
      }

      // Custom equation property
      VariableProperty property;

      // Whether the node is visible or has been erased
      // TODO: move to graph
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
    /**
     * Graph node representing an equation.
     *
     * Requires the underlying equation property to define the Id type and
     * implement the following methods:
     *  - Id getId() const : get the ID of the equation
     *  - size_t getNumOfIterationVars() : get the number of induction variables
     *  - Range getIterationRange(size_t inductionVarIndex) : get the range of
     *    an iteration variable.
     *  - void getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) : get
     *    the variable accesses done by this equation.
     */
    template<class EquationProperty, class VariableProperty>
    class EquationVertex : public Matchable
    {
      public:
      using Id = typename EquationProperty::Id;

      EquationVertex(EquationProperty property)
              : Matchable(IncidenceMatrix::column(getIterationRanges(property))),
                property(property),
                visible(true)
      {
      }

      void dump() const
      {
        dump(llvm::outs());
      }

      void dump(llvm::raw_ostream& os) const
      {
        os << "Equation\n";
        os << " |-- ID: " << getId() << "\n";
        os << " |-- Iteration ranges: ";

        for (size_t i = 0; i < getNumOfIterationVars(); ++i)
          os << getIterationRange(i);

        os << "\n";
        os << " |-- Match matrix:\n";
        os << getMatched();
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

      // TODO: replace unsigned int with size_t
      unsigned int getNumOfIterationVars() const
      {
        return getNumOfIterationVars(property);
      }

      Range getIterationRange(size_t index) const
      {
        return getIterationRange(property, index);
      }

      MultidimensionalRange getIterationRanges() const
      {
        return getIterationRanges(property);
      }

      unsigned int flatSize() const
      {
        return getIterationRanges().flatSize();
      }

      void getVariableAccesses(llvm::SmallVectorImpl<Access<VariableProperty>>& accesses) const
      {
        getVariableAccesses(property, accesses);
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
      static unsigned int getNumOfIterationVars(const EquationProperty& p)
      {
        return p.getNumOfIterationVars();
      }

      static Range getIterationRange(const EquationProperty& p, size_t index)
      {
        assert(index < getNumOfIterationVars(p));
        return p.getIterationRange(index);
      }

      static MultidimensionalRange getIterationRanges(const EquationProperty& p)
      {
        llvm::SmallVector<Range, 3> ranges;

        for (unsigned int i = 0, e = getNumOfIterationVars(p); i < e; ++i)
          ranges.push_back(getIterationRange(p, i));

        return MultidimensionalRange(ranges);
      }

      static void getVariableAccesses(
              const EquationProperty& p,
              llvm::SmallVectorImpl<Access<VariableProperty>>& accesses)
      {
        p.getVariableAccesses(accesses);
      }

      // Custom equation property
      EquationProperty property;

      // Whether the node is visible or has been erased
      // TODO: move to graph
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

      void dump() const
      {
        dump(llvm::outs());
      }

      void dump(llvm::raw_ostream& os) const
      {
        os << "Incidence matrix:\n" << incidenceMatrix << "\n";
        os << "Match matrix:\n" << matchMatrix;
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

      void removeMatch(IncidenceMatrix match)
      {
        matchMatrix -= match;
      }

      const IncidenceMatrix& getMatchMatrix() const
      {
        return matchMatrix;
      }

      IncidenceMatrix getUnmatchedMatrix() const
      {
        return incidenceMatrix - matchMatrix;
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

    template<typename VertexDescriptor, typename EdgeDescriptor>
    class AugmentingPath
    {
      public:
      class Flow
      {
        public:
        Flow(VertexDescriptor from, EdgeDescriptor edge, IncidenceMatrix delta)
              : from(std::move(from)), edge(std::move(edge)), delta(std::move(delta))
        {
        }

        VertexDescriptor getFrom() const
        {
          return from;
        }

        EdgeDescriptor getEdge() const
        {
          return edge;
        }

        const IncidenceMatrix& getDelta() const
        {
          return delta;
        }

        private:
        VertexDescriptor from;
        EdgeDescriptor edge;
        IncidenceMatrix delta;
      };

      private:
      template<typename T> using Container = llvm::SmallVector<T, 2>;

      public:
      using iterator = typename Container<Flow>::iterator;
      using const_iterator = typename Container<Flow>::const_iterator;

      AugmentingPath(llvm::ArrayRef<Flow> flows)
              : flows(flows.begin(), flows.end())
      {
      }

      const Flow& operator[](size_t index) const
      {
        assert(index < flows.size());
        return flows[index];
      }

      iterator begin()
      {
        return flows.begin();
      }

      const_iterator begin() const
      {
        return flows.begin();
      }

      iterator end()
      {
        return flows.end();
      }

      const_iterator end() const
      {
        return flows.end();
      }

      private:
      Container<Flow> flows;
    };

    /*
    template<typename VariableId, typename EquationId>
    class MatchingSolution
    {
      public:
      MatchingSolution(
              VariableId variable,
              EquationId equation,
              const detail::IncidenceMatrix* matchMatrix)
              : variable(std::move(variable)), equation(std::move(equation)), matchMatrix(matchMatrix)
      {
      }

      const VariableId& getVariable() const
      {
        return variable;
      }

      const EquationId& getEquation() const
      {
        return equation;
      }

      const detail::IncidenceMatrix* getMatchMatrix() const
      {
        return matchMatrix;
      }

      private:
      VariableId variable;
      EquationId equation;
      const detail::IncidenceMatrix* matchMatrix;
    };
     */
  }

  template<class VariableProperty, class EquationProperty>
  class MatchingGraph
  {
    public:
    using Variable = detail::VariableVertex<VariableProperty>;
    using Equation = detail::EquationVertex<EquationProperty, VariableProperty>;
    using Vertex = std::variant<Variable, Equation>;
    using Edge = detail::Edge;

    private:
    using Graph = base::Graph<Vertex, Edge>;

    using VertexDescriptor = typename Graph::VertexDescriptor;
    using EdgeDescriptor = typename Graph::EdgeDescriptor;

    using VertexIterator = typename Graph::VertexIterator;
    using EdgeIterator = typename Graph::EdgeIterator;
    using VisibleIncidentEdgeIterator = typename Graph::FilteredIncidentEdgeIterator;

    using AugmentingPath = detail::AugmentingPath<VertexDescriptor, EdgeDescriptor>;

    public:
    using VariableIterator = typename Graph::FilteredVertexIterator;
    using EquationIterator = typename Graph::FilteredVertexIterator;

    //using MatchingSolution = detail::MatchingSolution<typename Variable::Id, typename Equation::Id>;

    void dump() const;
    void dump(llvm::raw_ostream& os) const;

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

    llvm::iterator_range<VariableIterator> getVariables() const
    {
      auto filter = [&](const Vertex& vertex) -> bool {
          return std::holds_alternative<Variable>(vertex);
      };

      return graph.getVertices(filter);
    }

    void addVariable(VariableProperty property)
    {
      Variable variable(std::move(property));
      assert(!hasVariable(variable.getId()) && "Already existing variable");
      graph.addVertex(std::move(variable));
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

    llvm::iterator_range<EquationIterator> getEquations() const
    {
      auto filter = [&](const Vertex& vertex) -> bool {
          return std::holds_alternative<Equation>(vertex);
      };

      return graph.getVertices(filter);
    }

    void addEquation(EquationProperty property)
    {
      Equation equation(std::move(property));
      assert(!hasEquation(equation.getId()) && "Already existing equation");
      auto equationVertex = graph.addVertex(equation);

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
        auto edgeDescriptor = graph.addEdge(equationVertex, variableVertex, edge);

        graph[edgeDescriptor].addAccessFunction(access.getAccessFunction());
      }
    }

    size_t getNumberOfScalarVariables() const
    {
      size_t result = 0;

      for (const auto& v : graph.getVertices())
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

      for (const auto& v : graph.getVertices())
      {
        if (auto& vertex = graph[v]; std::holds_alternative<Equation>(vertex))
        {
          auto& equation = std::get<Equation>(vertex);
          result += equation.flatSize();
        }
      }

      return result;
    }

    bool allNodesMatched() const
    {
      auto vertices = graph.getVertices();

      auto allComponentsMatchedFn = [](const auto& obj) {
          return obj.allComponentsMatched();
      };

      return llvm::all_of(vertices, [&](const auto& descriptor) {
          return std::visit(allComponentsMatchedFn, graph[descriptor]);
      });
    }

    bool hasEdge(typename EquationProperty::Id equationId, typename VariableProperty::Id variableId) const
    {
      if (findEdge<Equation, Variable>(equationId, variableId).first)
        return true;

      return findEdge<Variable, Equation>(variableId, equationId).first;
    }

    EdgeDescriptor getFirstOutVisibleEdge(VertexDescriptor vertex) const
    {
      auto edges = getVisibleEdges(vertex);
      assert(edges.begin() != edges.end() && "Vertex doesn't belong to any edge");
      return *edges.begin();
    }

    bool simplify();

    bool match();

    private:
    template<typename T>
    bool hasVertex(typename T::Id id) const
    {
      return findVertex<T>(id).first;
    }

    template<typename T>
    std::pair<bool, VertexIterator> findVertex(typename T::Id id) const
    {
      auto vertices = graph.getVertices();

      auto it = std::find_if(vertices.begin(), vertices.end(), [&](const VertexDescriptor& v) {
        const auto& vertex = graph[v];

        if (!std::holds_alternative<T>(vertex))
          return false;

        return std::get<T>(vertex).getId() == id;
      });

      return std::make_pair(it != vertices.end(), it);
    }

    size_t getVertexVisibilityDegree(VertexDescriptor vertex) const
    {
      size_t result = 0;

      for (auto edge : getVisibleEdges(vertex))
        ++result;

      return result;
    }

    void remove(VertexDescriptor vertex)
    {
      std::visit([](auto& obj) -> void {
          obj.setVisibility(false);
      }, graph[vertex]);
    }

    template<typename From, typename To>
    std::pair<bool, EdgeIterator> findEdge(typename From::Id from, typename To::Id to) const
    {
      auto edges = graph.getEdges();

      auto it = std::find_if(edges.begin(), edges.end(), [&](const EdgeDescriptor& e) {
        auto& source = graph[e.from];
        auto& target = graph[e.to];

        if (!std::holds_alternative<From>(source) || !std::holds_alternative<To>(target))
          return false;

        return std::get<From>(source).getId() == from && std::get<To>(target).getId() == to;
      });

      return std::make_pair(it != edges.end(), it);
    }

    auto getEdges(VertexDescriptor vertex) const
    {
      return graph.getIncidentEdges(vertex);
    }

    llvm::iterator_range<VisibleIncidentEdgeIterator> getVisibleEdges(VertexDescriptor vertex) const
    {
      auto filter = [&](const Edge& edge) -> bool {
          return edge.isVisible();
      };

      return graph.getIncidentEdges(vertex, filter);
    }

    void remove(EdgeDescriptor edge)
    {
      graph[edge].setVisibility(false);
    }

    bool matchIteration();

    void getAugmentingPaths(llvm::SmallVectorImpl<AugmentingPath>& paths) const;

    void applyPath(const AugmentingPath& path);

    Graph graph;
  };

  template<typename VariableProperty, typename EquationProperty>
  void MatchingGraph<VariableProperty, EquationProperty>::dump() const
  {
    dump(llvm::outs());
  }

  template<typename VariableProperty, typename EquationProperty>
  void MatchingGraph<VariableProperty, EquationProperty>::dump(llvm::raw_ostream& os) const
  {
    os << "********** Matching graph **********\n";

    for (auto descriptor : graph.getVertices())
    {
      std::visit([&](const auto& vertex) {
        vertex.dump(os);
      }, graph[descriptor]);

      os << "\n";
    }

    for (auto descriptor : graph.getEdges())
    {
      const auto& from = std::visit([](const auto& vertex) {
        return vertex.getId();
      }, graph[descriptor.from]);

      const auto& to = std::visit([](const auto& vertex) {
          return vertex.getId();
      }, graph[descriptor.to]);

      os << "Edge from " << from << " to " << to << "\n";
      graph[descriptor].dump(os);
      os << "\n";
    }
  }

  template<typename VariableProperty, typename EquationProperty>
  bool MatchingGraph<VariableProperty, EquationProperty>::simplify()
  {
    // Vertices that are candidate for the first simplification phase.
    // They are the ones having only one incident edge.
    std::list<VertexDescriptor> candidates;

    for (VertexDescriptor vertex : graph.getVertices())
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

      auto edgeDescriptor = getFirstOutVisibleEdge(v1);
      Edge& edge = graph[edgeDescriptor];

      VertexDescriptor v2 = edgeDescriptor.from == v1 ? edgeDescriptor.to : edgeDescriptor.from;

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

        variable.addMatch(matchOptions[0].flattenEquations());
        equation.addMatch(matchOptions[0].flattenVariables());

        auto allComponentsMatchedVisitor = [](const auto& vertex) -> bool {
            return vertex.allComponentsMatched();
        };

        if (!std::visit(allComponentsMatchedVisitor, graph[v1]))
          return false;

        bool shouldRemoveOppositeNode = std::visit(allComponentsMatchedVisitor, graph[v2]);

        // Hide the edge
        remove(edgeDescriptor);

        // Hide the v1 vertex
        remove(v1);

        if (shouldRemoveOppositeNode)
        {
          for (auto e : graph.getIncidentEdges(v2))
          {
            remove(e);

            VertexDescriptor v = e.from == v2 ? e.to : e.from;

            auto isOtherVertexVisibleFn = [](const auto& obj) -> bool {
                return obj.isVisible();
            };

            if (!std::visit(isOtherVertexVisibleFn, graph[v]))
              continue;

            if (getVertexVisibilityDegree(v) == 1)
              candidates.push_back(v);
          }

          // Hide the v2 vertex and remove it from the candidates
          remove(v2);

          candidates.remove_if([&](auto v) {
              return v == v2;
          });
        }
        else
        {
          if (getVertexVisibilityDegree(v2) == 1)
            candidates.push_back(v2);
        }
      }
    }

    return true;
  }

  template<typename VariableProperty, typename EquationProperty>
  bool MatchingGraph<VariableProperty, EquationProperty>::match()
  {
    bool success;
    bool complete;

    do {
      success = matchIteration();
      complete = allNodesMatched();
    } while(success && !complete);

    return success;
  }

  template<typename VariableProperty, typename EquationProperty>
  bool MatchingGraph<VariableProperty, EquationProperty>::matchIteration()
  {
    llvm::SmallVector<AugmentingPath, 8> augmentingPaths;
    getAugmentingPaths(augmentingPaths);

    if (augmentingPaths.empty())
      return false;

    /*
    llvm::errs() << "---------------------------- AUGMENTING PATHS\n";

    for (const auto& path : augmentingPaths)
    {
      auto& os = llvm::errs();
      os << "---------AUGMENTING PATH\n";

      auto idVisitor = [](const auto& obj) { return obj.getId(); };

      for (const auto& node : path)
      {
        os << "--- NODE\n";
        os << "from " << std::visit(idVisitor, graph[node.getFrom()]) << "\n";
        os << "edge (" << std::visit(idVisitor, graph[node.getEdge().from]) << "," <<
           std::visit(idVisitor, graph[node.getEdge().to]) << ")\n";
        os << "delta\n";
        node.getDelta().dump();
        os << "\n";
      }
    }
     */

    for (auto& path : augmentingPaths)
      applyPath(path);

    return true;
  }

  namespace detail
  {
    template<typename VertexDescriptor, typename EdgeDescriptor>
    class Frontier
    {
      private:
      template<typename T> using Container = llvm::SmallVector<T, 10>;

      public:
      class BFSStep
      {
        public:
        BFSStep(VertexDescriptor node, IncidenceMatrix cur)
                : previous(nullptr),
                  node(std::move(node)),
                  cur(std::move(cur)),
                  edge(llvm::None),
                  match(llvm::None)
        {
        }

        BFSStep(BFSStep previous, EdgeDescriptor edge, VertexDescriptor node, IncidenceMatrix cur, IncidenceMatrix match)
                : previous(std::make_unique<BFSStep>(std::move(previous))),
                  node(std::move(node)),
                  cur(std::move(cur)),
                  edge(std::move(edge)),
                  match(std::move(match))
        {
        }

        BFSStep(const BFSStep& other)
                : previous(other.hasPrevious() ? std::make_unique<BFSStep>(*other.previous) : nullptr),
                  node(other.node),
                  cur(other.cur),
                  edge(other.edge),
                  match(other.match)
        {
        }

        BFSStep(BFSStep&& other) = default;

        ~BFSStep() = default;

        friend void swap(BFSStep& first, BFSStep& second)
        {
          using std::swap;

          swap(first.previous, second.previous);
          swap(first.node, second.node);
          swap(first.cur, second.cur);
        }

        bool hasPrevious() const
        {
          return previous.get() != nullptr;
        }

        const BFSStep& getPrevious() const
        {
          return *previous;
        }

        VertexDescriptor getNode() const
        {
          return node;
        }

        const IncidenceMatrix& curSet() const
        {
          return cur;
        }

        EdgeDescriptor getEdge() const
        {
          assert(edge.hasValue());
          return *edge;
        }

        const IncidenceMatrix& mapSet() const
        {
          assert(match.hasValue());
          return *match;
        }

        private:
        std::unique_ptr<BFSStep> previous;
        VertexDescriptor node;
        IncidenceMatrix cur;
        llvm::Optional<EdgeDescriptor> edge;
        llvm::Optional<IncidenceMatrix> match;
      };

      using iterator = typename Container<BFSStep>::iterator;
      using const_iterator = typename Container<BFSStep>::const_iterator;

      BFSStep& operator[](size_t index)
      {
        assert(index < steps.size());
        return steps[index];
      }

      const BFSStep& operator[](size_t index) const
      {
        assert(index < steps.size());
        return steps[index];
      }

      template<typename... Args>
      void add(Args&&... args)
      {
        steps.emplace_back(args...);
      }

      void clear()
      {
        steps.clear();
      }

      private:
      Container<BFSStep> steps;
    };
  }

  template<typename T>
  void insertOrAdd(std::map<T, detail::IncidenceMatrix>& map, T key, detail::IncidenceMatrix value)
  {
    if (auto it = map.find(key); it != map.end())
      it->second += value;
    else
      map.emplace(key, std::move(value));
  }
}

namespace marco::matching {
  template<typename VariableProperty, typename EquationProperty>
  void MatchingGraph<VariableProperty, EquationProperty>::getAugmentingPaths(
          llvm::SmallVectorImpl<AugmentingPath>& paths) const
  {
    using Frontier = detail::Frontier<VertexDescriptor, EdgeDescriptor>;
    using BFSStep = typename Frontier::BFSStep;

    std::vector<BFSStep> frontier;

    // Calculation of the initial frontier
    auto equations = getEquations();

    for (auto equationDescriptor : equations)
    {
      const Equation& equation = getEquation(equationDescriptor);

      if (auto unmatchedEquations = equation.getUnmatched(); !unmatchedEquations.isEmpty())
        frontier.emplace_back(equationDescriptor, std::move(unmatchedEquations));
    }

    // Breadth-first search
    std::vector<BFSStep> newFrontier;
    llvm::SmallVector<BFSStep, 10> foundPaths;

    while (!frontier.empty() && foundPaths.empty())
    {
      /*
      llvm::errs() << "----------------------- FRONTIER\n";
      for (const auto& step : frontier)
      {
        auto& os = llvm::errs();
        os << "--------- STEP\n";

        auto idVisitor = [](const auto& obj) { return obj.getId(); };

        os << "node: " << std::visit(idVisitor, graph[step.getNode()]) << "\n";

        if (step.hasPrevious())
        {
          os << "from " << std::visit(idVisitor, graph[step.getEdge().from]);
          os << " to " << std::visit(idVisitor, graph[step.getEdge().to]) << "\n";
        }

        os << "curSet\n";
        step.curSet().dump();

        if (step.hasPrevious())
        {
          os << "mapSet\n";
          step.mapSet().dump();
        }
      }
      */

      for (BFSStep& step : frontier)
      {
        auto vertexDescriptor = step.getNode();

        for (EdgeDescriptor edgeDescriptor : getEdges(vertexDescriptor))
        {
          assert(edgeDescriptor.from == vertexDescriptor);
          VertexDescriptor nextNode = edgeDescriptor.to;
          const Edge& edge = graph[edgeDescriptor];

          if (isEquation(vertexDescriptor))
          {
            assert(isVariable(nextNode));
            const Equation& equation = getEquation(vertexDescriptor);
            const Variable& variable = getVariable(nextNode);
            auto unmatchedMatrix = edge.getUnmatchedMatrix();
            auto filteredMatrix = unmatchedMatrix.filterEquations(step.curSet());
            detail::LocalMatchingSolutions solutions = detail::solveLocalMatchingProblem(filteredMatrix);

            for (auto solution : solutions)
            {
              Variable var = getVariable(edgeDescriptor.to);
              auto unmatchedScalarVariables = var.getUnmatched();
              auto matched = solution.filterVariables(unmatchedScalarVariables);

              if (!matched.isEmpty())
              {
                foundPaths.emplace_back(step, edgeDescriptor, nextNode, matched.flattenEquations(), matched);
              }
              else
              {
                newFrontier.emplace_back(step, edgeDescriptor, nextNode, solution.flattenEquations(), solution);
              }
            }
          }
          else
          {
            assert(isEquation(nextNode));
            auto filteredMatrix = edge.getMatchMatrix().filterVariables(step.curSet());
            detail::LocalMatchingSolutions solutions = detail::solveLocalMatchingProblem(filteredMatrix);

            for (auto solution : solutions)
              newFrontier.emplace_back(BFSStep(step, edgeDescriptor, nextNode, solution.flattenVariables(), solution));
          }
        }
      }

      frontier.clear();
      frontier.swap(newFrontier);
    }

    /*
    llvm::errs() << "----------------------- FOUND PATHS\n";
    for (const auto& pathEnd : foundPaths)
    {
      auto& os = llvm::errs();
      os << "--------- PATH\n";

      const BFSStep* step = &pathEnd;

      while (step)
      {
        os << "--------- NODE\n";
        auto idVisitor = [](const auto& obj) { return obj.getId(); };

        os << "node: " << std::visit(idVisitor, graph[step->getNode()]) << "\n";

        if (step->hasPrevious())
        {
          os << "from " << std::visit(idVisitor, graph[step->getEdge().from]);
          os << " to " << std::visit(idVisitor, graph[step->getEdge().to]) << "\n";
        }

        os << "curSet\n";
        step->curSet().dump();

        if (step->hasPrevious())
        {
          os << "mapSet\n";
          step->mapSet().dump();
        }

        step = &step->getPrevious();
      }
    }
     */

    // For each traversed node, keep track of the indexes that have already
    // been traversed by some augmenting path. A new candidate path can be
    // accepted only if it does not traverse any of them.
    std::map<VertexDescriptor, detail::IncidenceMatrix> visited;

    for (BFSStep& pathEnd : foundPaths)
    {
      // All the candidate paths consist in at least two nodes by construction
      assert(pathEnd.hasPrevious());

      std::vector<typename AugmentingPath::Flow> flows;

      // The path's validity is unknown, so we must avoid polluting the
      // list of visited scalar nodes. If the path will be marked as valid,
      // then the new visits will be merged with the already found ones.
      std::map<VertexDescriptor, detail::IncidenceMatrix> newVisits;

      const BFSStep* curStep = &pathEnd;
      detail::IncidenceMatrix map = curStep->mapSet();
      bool validPath = true;

      while (curStep && validPath)
      {
        if (curStep->hasPrevious())
        {
          if (!flows.empty())
          {
            auto prevMap = flows[0].getDelta();

            if (isVariable(curStep->getNode()))
              map = curStep->mapSet().filterVariables(prevMap.flattenEquations());
            else
              map = curStep->mapSet().filterEquations(prevMap.flattenVariables());
          }

          flows.emplace(flows.begin(), curStep->getPrevious().getNode(), curStep->getEdge(), map);
        }

        auto touchedIndexes = isVariable(curStep->getNode()) ? map.flattenEquations() : map.flattenVariables();

        if (auto it = visited.find(curStep->getNode()); it != visited.end())
        {
          auto& alreadyTouchedIndices = it->second;

          if (!touchedIndexes.isDisjoint(alreadyTouchedIndices))
          {
            // The current path intersects another one, so we need to discard it.
            validPath = false;
          }
          else
          {
            insertOrAdd(newVisits, curStep->getNode(), alreadyTouchedIndices + touchedIndexes);
          }
        }
        else
        {
          insertOrAdd(newVisits, curStep->getNode(), touchedIndexes);
        }

        // Move backwards inside the candidate augmenting path
        curStep = &curStep->getPrevious();
      }

      if (validPath)
      {
        paths.emplace_back(flows);

        for (auto& p : newVisits)
          visited.insert_or_assign(p.first, p.second);
      }
    }
  }

  template<typename VariableProperty, typename EquationProperty>
  void MatchingGraph<VariableProperty, EquationProperty>::applyPath(const AugmentingPath& path)
  {
    std::map<VertexDescriptor, detail::IncidenceMatrix> removedMatches;
    std::map<VertexDescriptor, detail::IncidenceMatrix> newMatches;

    for (auto& flow : path)
    {
      assert(flow.getFrom() == flow.getEdge().from || flow.getFrom() == flow.getEdge().to);
      auto& edge = graph[flow.getEdge()];

      VertexDescriptor from = flow.getFrom();
      VertexDescriptor to = flow.getEdge().from == from ? flow.getEdge().to : flow.getEdge().from;

      auto& delta = flow.getDelta();
      auto deltaEquations = delta.flattenVariables();
      auto deltaVariables = delta.flattenEquations();

      if (isVariable(from))
      {
        insertOrAdd(removedMatches, from, deltaVariables);
        insertOrAdd(removedMatches, to, deltaEquations);
        edge.removeMatch(delta);
      }
      else
      {
        insertOrAdd(newMatches, from, deltaEquations);
        insertOrAdd(newMatches, to, deltaVariables);
        edge.addMatch(delta);
      }
    }

    for (const auto& match : removedMatches)
    {
      std::visit([&match](auto& node) {
        node.removeMatch(match.second);
      }, graph[match.first]);
    }

    for (const auto& match : newMatches)
    {
      std::visit([&match](auto& node) {
          node.addMatch(match.second);
      }, graph[match.first]);
    }
  }
}

#endif	// MARCO_MATCHING_MATCHING_H
