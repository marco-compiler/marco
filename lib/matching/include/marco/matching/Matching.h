#ifndef MARCO_MATCHING_MATCHING_H
#define MARCO_MATCHING_MATCHING_H

#include <iostream>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <marco/utils/TreeOStream.h>
#include <memory>
#include <numeric>
#include <variant>

#include "AccessFunction.h"
#include "Dumpable.h"
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
       * Check whether all the scalar elements of this array have
       * been matched.
       *
       * @return true if all the elements are matched
       */
      bool allComponentsMatched() const;

      void addMatch(const IncidenceMatrix& newMatch);
      void removeMatch(const IncidenceMatrix& removedMatch);

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
    class VariableVertex : public Matchable, public Dumpable
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

      using Dumpable::dump;

      void dump(std::ostream& stream) const override
      {
        using namespace marco::utils;

        TreeOStream os(stream);
        os << "Variable\n";
        os << tree_property << "ID: " << getId() << "\n";
        os << tree_property << "Rank: " << getRank() << "\n";
        os << tree_property << "Dimensions: ";

        for (size_t i = 0; i < getRank(); ++i)
          os << "[" << getDimensionSize(i) << "]";

        os << "\n";
        os << tree_property << "Match matrix:\n" << getMatched();

        stream << std::endl;
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

      size_t getRank() const
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
      static size_t getRank(const VariableProperty& p)
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
    class EquationVertex : public Matchable, public Dumpable
    {
      public:
      using Id = typename EquationProperty::Id;

      EquationVertex(EquationProperty property)
              : Matchable(IncidenceMatrix::column(getIterationRanges(property))),
                property(property),
                visible(true)
      {
      }

      using Dumpable::dump;

      void dump(std::ostream& stream) const override
      {
        using namespace marco::utils;

        TreeOStream os(stream);
        os << "Equation\n";
        os << tree_property << "ID: " << getId() << "\n";
        os << tree_property << "Iteration ranges: ";

        for (size_t i = 0; i < getNumOfIterationVars(); ++i)
          os << getIterationRange(i);

        os << "\n";
        os << tree_property << "Match matrix:\n" << getMatched();

        stream << std::endl;
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

      size_t getNumOfIterationVars() const
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
      static size_t getNumOfIterationVars(const EquationProperty& p)
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

    template<typename Variable, typename Equation>
    class Edge : public Dumpable
    {
      public:
      Edge(typename Equation::Id equation,
           typename Variable::Id variable,
           MultidimensionalRange equationRanges,
           MultidimensionalRange variableRanges)
              : equation(std::move(equation)),
                variable(std::move(variable)),
                incidenceMatrix(equationRanges, variableRanges),
                matchMatrix(equationRanges, variableRanges),
                visible(true)
      {
      }

      using Dumpable::dump;

      void dump(std::ostream& stream) const override
      {
        using namespace marco::utils;

        TreeOStream os(stream);
        os << "Edge\n";
        os << tree_property << "Equation: " << equation << "\n";
        os << tree_property << "Variable: " << variable << "\n";
        os << tree_property << "Incidence matrix:\n" << incidenceMatrix << "\n";
        os << tree_property << "Match matrix:\n" << getMatched();

        stream << std::endl;
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

      const IncidenceMatrix& getMatched() const
      {
        return matchMatrix;
      }

      IncidenceMatrix getUnmatched() const
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
      // Equation's ID. Just for debugging purpose
      typename Equation::Id equation;

      // Variable's ID. Just for debugging purpose
      typename Variable::Id variable;

      llvm::SmallVector<AccessFunction, 3> accessFunctions;
      IncidenceMatrix incidenceMatrix;
      IncidenceMatrix matchMatrix;

      // TODO: move to graph
      bool visible;
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

  template<typename VariableProperty, typename EquationProperty>
  class MatchingGraph : public detail::Dumpable
  {
    public:
    using Variable = detail::VariableVertex<VariableProperty>;
    using Equation = detail::EquationVertex<EquationProperty, VariableProperty>;
    using Vertex = std::variant<Variable, Equation>;
    using Edge = detail::Edge<Variable, Equation>;

    private:
    using Graph = base::Graph<Vertex, Edge>;

    using VertexDescriptor = typename Graph::VertexDescriptor;
    using EdgeDescriptor = typename Graph::EdgeDescriptor;

    using VertexIterator = typename Graph::VertexIterator;
    using EdgeIterator = typename Graph::EdgeIterator;
    using VisibleIncidentEdgeIterator = typename Graph::FilteredIncidentEdgeIterator;

    using IncidenceMatrix = detail::IncidenceMatrix;
    class Frontier;
    class AugmentingPath;

    public:
    using VariableIterator = typename Graph::FilteredVertexIterator;
    using EquationIterator = typename Graph::FilteredVertexIterator;

    //using MatchingSolution = detail::MatchingSolution<typename Variable::Id, typename Equation::Id>;

    using Dumpable::dump;
    void dump(std::ostream& os) const override;

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

    Variable& getVariable(VertexDescriptor descriptor)
    {
      auto& vertex = graph[descriptor];
      assert(std::holds_alternative<Variable>(vertex));
      return std::get<Variable>(vertex);
    }

    const Variable& getVariable(VertexDescriptor descriptor) const
    {
      auto& vertex = graph[descriptor];
      assert(std::holds_alternative<Variable>(vertex));
      return std::get<Variable>(vertex);
    }

    llvm::iterator_range<VariableIterator> getVariables() const
    {
      auto filter = [](const Vertex& vertex) -> bool {
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

    Equation& getEquation(VertexDescriptor descriptor)
    {
      auto& vertex = graph[descriptor];
      assert(std::holds_alternative<Equation>(vertex));
      return std::get<Equation>(vertex);
    }

    const Equation& getEquation(VertexDescriptor descriptor) const
    {
      auto& vertex = graph[descriptor];
      assert(std::holds_alternative<Equation>(vertex));
      return std::get<Equation>(vertex);
    }

    llvm::iterator_range<EquationIterator> getEquations() const
    {
      auto filter = [](const Vertex& vertex) -> bool {
          return std::holds_alternative<Equation>(vertex);
      };

      return graph.getVertices(filter);
    }

    void addEquation(EquationProperty property)
    {
      Equation eq(std::move(property));
      assert(!hasEquation(eq.getId()) && "Already existing equation");

      // Insert the equation into the graph and get a reference to the new vertex
      auto equationDescriptor = graph.addVertex(std::move(eq));
      Equation& equation = getEquation(equationDescriptor);

      llvm::SmallVector<Access<VariableProperty>, 3> accesses;
      equation.getVariableAccesses(accesses);

      // The equation may access multiple variables or even multiple indexes
      // of the same variable. Add an edge to the graph for each of those
      // accesses.

      for (const auto& access : accesses)
      {
        auto variableDescriptor = getVariableVertex(access.getVariable().getId());
        Variable& variable = getVariable(variableDescriptor);

        Edge edge(equation.getId(), variable.getId(), equation.getIterationRanges(), variable.getRanges());
        auto edgeDescriptor = graph.addEdge(equationDescriptor, variableDescriptor, edge);
        graph[edgeDescriptor].addAccessFunction(access.getAccessFunction());
      }
    }

    /**
     * Get the total amount of scalar variables inside the graph.
     *
     * @return number of scalar variables
     */
    size_t getNumberOfScalarVariables() const
    {
      auto vars = getVariables();

      return std::accumulate(vars.begin(), vars.end(), 0, [&](size_t sum, const auto& desc) {
        return sum + getVariable(desc).flatSize();
      });
    }

    /**
     * Get the total amount of scalar equations inside the graph.
     * With "scalar equations" we mean the ones generated by unrolling
     * the loops defining them.
     *
     * @return number of scalar equations
     */
    size_t getNumberOfScalarEquations() const
    {
      auto eqs = getEquations();

      return std::accumulate(eqs.begin(), eqs.end(), 0, [&](size_t sum, const auto& desc) {
          return sum + getEquation(desc).flatSize();
      });
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

    /**
     * Apply the simplification algorithm in order to perform all
     * the obligatory matches, that is the variables and equations
     * having only one incident edge.
     *
     * @return true if the simplification algorithm didn't find any inconsistency
     */
    bool simplify();

    /**
     * Apply the matching algorithm.
     *
     * @return true if the matching algorithm managed to fully match all the nodes
     */
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
      auto edges = getVisibleEdges(vertex);
      return std::distance(edges.begin(), edges.end());
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

    /**
     * Apply an augmenting path to the graph.
     *
     * @param path augmenting path
     */
    void applyPath(const AugmentingPath& path);

    Graph graph;
  };

  template<typename VariableProperty, typename EquationProperty>
  class MatchingGraph<VariableProperty, EquationProperty>::Frontier : public detail::Dumpable
  {
    private:
    template<typename T> using Container = std::vector<T>;

    public:
    class BFSStep : public Dumpable
    {
      public:
      BFSStep(const Graph& graph,
              VertexDescriptor node,
              IncidenceMatrix candidates)
              : graph(&graph),
                previous(nullptr),
                node(std::move(node)),
                candidates(std::move(candidates)),
                edge(llvm::None),
                mappedFlow(llvm::None)
      {
      }

      BFSStep(const Graph& graph,
              BFSStep previous,
              EdgeDescriptor edge,
              VertexDescriptor node,
              IncidenceMatrix candidates,
              IncidenceMatrix mappedFlow)
              : graph(&graph),
                previous(std::make_unique<BFSStep>(std::move(previous))),
                node(std::move(node)),
                candidates(std::move(candidates)),
                edge(std::move(edge)),
                mappedFlow(std::move(mappedFlow))
      {
      }

      BFSStep(const BFSStep& other)
              : graph(other.graph),
                previous(other.hasPrevious() ? std::make_unique<BFSStep>(*other.previous) : nullptr),
                node(other.node),
                candidates(other.candidates),
                edge(other.edge),
                mappedFlow(other.mappedFlow)
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

      using Dumpable::dump;

      void dump(std::ostream& stream) const override
      {
        using namespace marco::utils;

        TreeOStream os(stream);
        os << "BFS step\n";

        auto idVisitor = [](const auto& vertex) { return vertex.getId(); };

        os << tree_property << "Node: " << std::visit(idVisitor, (*graph)[getNode()]) << "\n";
        os << tree_property << "Candidates:\n" << getCandidates();

        if (hasPrevious())
        {
          os << "\n";
          os << tree_property << "Edge: "
             << std::visit(idVisitor, (*graph)[getEdge().from])
             << " - "
             << std::visit(idVisitor, (*graph)[getEdge().to])
             << "\n";

          os << tree_property << "Mapped flow:\n" << getMappedFlow() << "\n";

          os << tree_property << "Previous:\n";
          getPrevious().dump(os);
        }
      }

      bool hasPrevious() const
      {
        return previous.get() != nullptr;
      }

      const BFSStep& getPrevious() const
      {
        return *previous;
      }

      const VertexDescriptor& getNode() const
      {
        return node;
      }

      const IncidenceMatrix& getCandidates() const
      {
        return candidates;
      }

      const EdgeDescriptor& getEdge() const
      {
        assert(edge.hasValue());
        return *edge;
      }

      const IncidenceMatrix& getMappedFlow() const
      {
        assert(mappedFlow.hasValue());
        return *mappedFlow;
      }

      private:
      // Stored for debugging purpose
      const Graph* graph;

      std::unique_ptr<BFSStep> previous;
      VertexDescriptor node;
      IncidenceMatrix candidates;
      llvm::Optional<EdgeDescriptor> edge;
      llvm::Optional<IncidenceMatrix> mappedFlow;
    };

    using iterator = typename Container<BFSStep>::iterator;
    using const_iterator = typename Container<BFSStep>::const_iterator;

    using Dumpable::dump;

    void dump(std::ostream& stream) const override
    {
      using namespace marco::utils;

      TreeOStream os(stream);
      os << "Frontier\n";

      for (const auto& step : steps)
      {
        os << tree_property;
        step.dump(os);
        os << "\n";
      }
    }

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

    bool empty() const
    {
      return steps.empty();
    }

    template<typename... Args>
    void emplace(Args&&... args)
    {
      steps.emplace_back(args...);
    }

    void clear()
    {
      steps.clear();
    }

    void swap(Frontier& other)
    {
      steps.swap(other.steps);
    }

    const_iterator begin() const
    {
      return steps.begin();
    }

    const_iterator end() const
    {
      return steps.end();
    }

    private:
    Container<BFSStep> steps;
  };

  template<typename VariableProperty, typename EquationProperty>
  class MatchingGraph<VariableProperty, EquationProperty>::AugmentingPath : public detail::Dumpable
  {
    public:
    struct Flow : public detail::Dumpable
    {
      Flow(VertexDescriptor source, EdgeDescriptor edge, const IncidenceMatrix& delta, const Graph& graph)
              : source(std::move(source)),
                edge(std::move(edge)),
                delta(std::move(delta)),
                graph(&graph)
      {
        assert(this->source == this->edge.from || this->source == this->edge.to);
      }

      using Dumpable::dump;

      void dump(std::ostream& stream) const override
      {
        using namespace marco::utils;

        TreeOStream os(stream);
        os << "Flow\n";

        auto idVisitor = [](const auto& obj) { return obj.getId(); };

        os << tree_property << "Source: " << std::visit(idVisitor, (*graph)[source]) << "\n";
        os << tree_property << "Edge: ";
        os << std::visit(idVisitor, (*graph)[edge.from]) << " - " << std::visit(idVisitor, (*graph)[edge.to]) << "\n";
        os << tree_property << "Delta:\n" << delta;
      }

      const VertexDescriptor source;
      const EdgeDescriptor edge;
      const IncidenceMatrix delta;

      // Stored for debugging purpose
      const Graph* graph;
    };

    private:
    template<typename T> using Container = std::vector<T>;

    public:
    using const_iterator = typename Container<Flow>::const_iterator;

    template<typename Flows>
    AugmentingPath(const Flows& flows)
            : flows(flows.begin(), flows.end())
    {
    }

    using Dumpable::dump;

    void dump(std::ostream& stream) const override
    {
      using namespace marco::utils;

      TreeOStream os(stream);
      os << "Augmenting path\n";

      for (const auto& flow : flows)
      {
        os << tree_property;
        flow.dump(os);
        os << "\n";
      }
    }

    const Flow& operator[](size_t index) const
    {
      assert(index < flows.size());
      return flows[index];
    }

    const_iterator begin() const
    {
      return flows.begin();
    }

    const_iterator end() const
    {
      return flows.end();
    }

    private:
    Container<Flow> flows;
  };

  template<typename VariableProperty, typename EquationProperty>
  void MatchingGraph<VariableProperty, EquationProperty>::dump(std::ostream& stream) const
  {
    using namespace marco::utils;

    TreeOStream os(stream);
    os << "Matching graph\n";

    for (auto descriptor : graph.getVertices())
    {
      std::visit([&](const auto& vertex) {
        os << tree_property;
        vertex.dump(os);
      }, graph[descriptor]);
    }

    for (auto descriptor : graph.getEdges())
    {
      os << tree_property;
      graph[descriptor].dump(os);
    }
  }

  template<typename VariableProperty, typename EquationProperty>
  bool MatchingGraph<VariableProperty, EquationProperty>::simplify()
  {
    // Vertices that are candidate for the first simplification phase.
    // They are the ones having only one incident edge.
    std::list<VertexDescriptor> candidates;

    // Determine the initial set of vertices with exactly one incident edge
    for (VertexDescriptor vertex : graph.getVertices())
    {
      auto incidentEdges = getVertexVisibilityDegree(vertex);

      if (incidentEdges == 0)
        return false;

      if (incidentEdges == 1)
        candidates.push_back(vertex);
    }

    // Iterate on the candidate vertices and apply the simplification algorithm
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

        // Remove the edge and the current candidate vertex
        remove(edgeDescriptor);
        remove(v1);

        if (shouldRemoveOppositeNode)
        {
          // When a node is removed, then also its incident edges are
          // removed. This can lead to new obliged matches, like in the
          // following example:
          //        |-- v3 ----
          // v1 -- v2         |
          //        |-- v4 -- v5
          // v1 is the current candidate and thus is removed.
          // v2 is removed because fully matched.
          // v3 and v4 become new candidates for the simplification pass.

          for (auto e : graph.getIncidentEdges(v2))
          {
            remove(e);

            VertexDescriptor v = e.from == v2 ? e.to : e.from;

            // TODO: remove visibility filter once the visibility is moved to the base graph
            auto isOtherVertexVisibleFn = [](const auto& obj) -> bool {
                return obj.isVisible();
            };

            if (!std::visit(isOtherVertexVisibleFn, graph[v]))
              continue;

            if (getVertexVisibilityDegree(v) == 1)
              candidates.push_back(v);
          }

          // Remove the v2 vertex and remove it from the candidates
          remove(v2);

          candidates.remove_if([&](auto v) {
              return v == v2;
          });
        }
        else
        {
          // When an edge is removed but one its vertices survives, we must
          // check if the remaining vertex has an obliged match.

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

    for (auto& path : augmentingPaths)
      applyPath(path);

    return true;
  }

  namespace detail
  {
    template<typename T>
    void insertOrAdd(std::map<T, IncidenceMatrix>& map, T key, IncidenceMatrix value)
    {
      if (auto it = map.find(key); it != map.end())
        it->second += value;
      else
        map.emplace(key, std::move(value));
    }
  }

  template<typename VariableProperty, typename EquationProperty>
  void MatchingGraph<VariableProperty, EquationProperty>::getAugmentingPaths(
          llvm::SmallVectorImpl<AugmentingPath>& paths) const
  {
    using BFSStep = typename Frontier::BFSStep;

    Frontier frontier;

    // Calculation of the initial frontier
    auto equations = getEquations();

    for (auto equationDescriptor : equations)
    {
      const Equation& equation = getEquation(equationDescriptor);

      if (auto unmatchedEquations = equation.getUnmatched(); !unmatchedEquations.isEmpty())
        frontier.emplace(graph, equationDescriptor, std::move(unmatchedEquations));
    }

    // Breadth-first search
    Frontier newFrontier;
    llvm::SmallVector<BFSStep, 10> foundPaths;

    while (!frontier.empty() && foundPaths.empty())
    {
      for (const BFSStep& step : frontier)
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
            auto unmatchedMatrix = edge.getUnmatched();
            auto filteredMatrix = unmatchedMatrix.filterEquations(step.getCandidates());
            detail::LocalMatchingSolutions solutions = detail::solveLocalMatchingProblem(filteredMatrix);

            for (auto solution : solutions)
            {
              Variable var = getVariable(edgeDescriptor.to);
              auto unmatchedScalarVariables = var.getUnmatched();
              auto matched = solution.filterVariables(unmatchedScalarVariables);

              if (!matched.isEmpty())
                foundPaths.emplace_back(graph, step, edgeDescriptor, nextNode, matched.flattenEquations(), matched);
              else
                newFrontier.emplace(graph, step, edgeDescriptor, nextNode, solution.flattenEquations(), solution);
            }
          }
          else
          {
            assert(isEquation(nextNode));
            auto filteredMatrix = edge.getMatched().filterVariables(step.getCandidates());
            detail::LocalMatchingSolutions solutions = detail::solveLocalMatchingProblem(filteredMatrix);

            for (auto solution : solutions)
              newFrontier.emplace(graph, step, edgeDescriptor, nextNode, solution.flattenVariables(), solution);
          }
        }
      }

      // Set the new frontier for the next iteration
      frontier.clear();
      frontier.swap(newFrontier);
    }

    // For each traversed node, keep track of the indexes that have already
    // been traversed by some augmenting path. A new candidate path can be
    // accepted only if it does not traverse any of them.
    std::map<VertexDescriptor, IncidenceMatrix> visited;

    for (BFSStep& pathEnd : foundPaths)
    {
      // All the candidate paths consist in at least two nodes by construction
      assert(pathEnd.hasPrevious());

      std::list<typename AugmentingPath::Flow> flows;

      // The path's validity is unknown, so we must avoid polluting the
      // list of visited scalar nodes. If the path will be marked as valid,
      // then the new visits will be merged with the already found ones.
      std::map<VertexDescriptor, IncidenceMatrix> newVisits;

      const BFSStep* curStep = &pathEnd;
      IncidenceMatrix map = curStep->getMappedFlow();
      bool validPath = true;

      while (curStep && validPath)
      {
        if (curStep->hasPrevious())
        {
          if (!flows.empty())
          {
            // Restrict the flow
            const auto& prevMap = flows.front().delta;

            if (isVariable(curStep->getNode()))
              map = curStep->getMappedFlow().filterVariables(prevMap.flattenEquations());
            else
              map = curStep->getMappedFlow().filterEquations(prevMap.flattenVariables());
          }

          flows.emplace(flows.begin(), curStep->getPrevious().getNode(), curStep->getEdge(), map, graph);
        }

        auto touchedIndexes = isVariable(curStep->getNode()) ? map.flattenEquations() : map.flattenVariables();

        if (auto it = visited.find(curStep->getNode()); it != visited.end())
        {
          auto& alreadyTouchedIndices = it->second;

          if (!touchedIndexes.isDisjoint(alreadyTouchedIndices))
          {
            // The current path intersects another one, so we need to discard it
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
        paths.emplace_back(std::move(flows));

        for (auto& p : newVisits)
          visited.insert_or_assign(p.first, p.second);
      }
    }
  }

  template<typename VariableProperty, typename EquationProperty>
  void MatchingGraph<VariableProperty, EquationProperty>::applyPath(const AugmentingPath& path)
  {
    // In order to preserve consistency of the match information among
    // edges and nodes, we need to separately track the modifications
    // created by the augmenting path on the vertices and apply all the
    // removals before the additions.
    // Consider in fact the example path [eq1 -> x -> eq2]: the first
    // move would add some match information to eq1 and x, while the
    // subsequent x -> eq2 would remove some from x. However, being the
    // match matrices made of booleans, the components of x that are
    // matched by eq1 would result as unmatched. If we instead first
    // apply the removals, the new matches are not wrongly erased anymore.

    std::map<VertexDescriptor, IncidenceMatrix> removedMatches;
    std::map<VertexDescriptor, IncidenceMatrix> newMatches;

    // Update the match matrices on the edges and store the information
    // about the vertices to be updated later.

    for (auto& flow : path)
    {
      auto& edge = graph[flow.edge];

      VertexDescriptor from = flow.source;
      VertexDescriptor to = flow.edge.from == from ? flow.edge.to : flow.edge.from;

      auto deltaEquations = flow.delta.flattenVariables();
      auto deltaVariables = flow.delta.flattenEquations();

      if (isVariable(from))
      {
        // Backward node
        insertOrAdd(removedMatches, from, deltaVariables);
        insertOrAdd(removedMatches, to, deltaEquations);
        edge.removeMatch(flow.delta);
      }
      else
      {
        // Forward node
        insertOrAdd(newMatches, from, deltaEquations);
        insertOrAdd(newMatches, to, deltaVariables);
        edge.addMatch(flow.delta);
      }
    }

    // Update the match information stored on the vertices

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
