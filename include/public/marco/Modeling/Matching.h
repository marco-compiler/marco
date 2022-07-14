#ifndef MARCO_MODELING_MATCHING_H
#define MARCO_MODELING_MATCHING_H

#include "marco/Diagnostic/TreeOStream.h"
#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/Graph.h"
#include "marco/Modeling/LocalMatchingSolutions.h"
#include "marco/Modeling/MCIM.h"
#include "marco/Modeling/Range.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <type_traits>
#include <variant>

namespace marco::modeling
{
  namespace matching
  {
    // This class must be specialized for the variable type that is used during the matching process.
    template<typename VariableType>
    struct VariableTraits
    {
      // Elements to provide:
      //
      // typedef Id : the ID type of the variable.
      //
      // static Id getId(const VariableType*)
      //    return the ID of the variable.
      //
      // static size_t getRank(const VariableType*)
      //    return the number of dimensions.
      //
      // static IndexSet getIndices(const VariableType*)
      //    return the indices of a variable.

      using Id = typename VariableType::UnknownVariableTypeError;
    };

    // This class must be specialized for the equation type that is used during the matching process.
    template<typename EquationType>
    struct EquationTraits
    {
      // Elements to provide:
      //
      // typedef Id : the ID type of the equation.
      //
      // static Id getId(const EquationType*)
      //    return the ID of the equation.
      //
      // static size_t getNumOfIterationVars(const EquationType*)
      //    return the number of induction variables.
      //
      // static MultidimensionalRange getIterationRanges(const EquationType*)
      //    return the iteration ranges.
      //
      // typedef VariableType : the type of the accessed variable
      //
      // typedef AccessProperty : the access property (this is optional, and if not specified an empty one is used)
      //
      // static std::vector<Access<VariableType, AccessProperty>> getAccesses(const EquationType*)
      //    return the access done by the equation.

      using Id = typename EquationType::UnknownEquationTypeError;
    };
  }

  namespace internal
  {
    namespace matching
    {
      /// Represent a generic vectorized entity whose scalar elements
      /// can be matched with the scalar elements of other arrays.
      /// The relationship is tracked by means of an incidence matrix.
      class Matchable
      {
        public:
          Matchable(IndexSet matchableIndices);

          const IndexSet& getMatched() const;

          IndexSet getUnmatched() const;

          /// Check whether all the scalar elements of this array have been matched.
          bool allComponentsMatched() const;

          void addMatch(const IndexSet& newMatch);

          void removeMatch(const IndexSet& removedMatch);

        private:
          IndexSet matchableIndices;
          IndexSet match;
      };

      /// Graph node representing a variable.
      template<typename VariableProperty>
      class VariableVertex : public Matchable, public Dumpable
      {
        public:
          using Property = VariableProperty;
          using Traits = typename ::marco::modeling::matching::VariableTraits<VariableProperty>;
          using Id = typename Traits::Id;

          VariableVertex(VariableProperty property)
              : Matchable(getIndices(property)),
                property(property),
                visible(true)
          {
            // Scalar variables can be represented by means of an array with just one element
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
            os << tree_property << "Indices: " << getIndices() << "\n";
            os << tree_property << "Matched: " << getMatched() << "\n";

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

          Id getId() const
          {
            return Traits::getId(&property);
          }

          size_t getRank() const
          {
            return getRank(property);
          }

          IndexSet getIndices() const
          {
            return getIndices(property);
          }

          unsigned int flatSize() const
          {
            return getIndices().flatSize();
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
            return Traits::getRank(&p);
          }

          static IndexSet getIndices(const VariableProperty& p)
          {
            return Traits::getIndices(&p);
          }

          // Custom equation property
          VariableProperty property;

          // Whether the node is visible or has been erased
          // TODO: move to graph
          bool visible;
      };

      class EmptyAccessProperty
      {
      };

      template<class EquationProperty>
      struct get_access_property
      {
        template<typename U>
        using Traits = ::marco::modeling::matching::EquationTraits<U>;

        template<class U, typename = typename Traits<U>::AccessProperty>
        static typename Traits<U>::AccessProperty property(int);

        template<class U>
        static EmptyAccessProperty property(...);

        using type = decltype(property<EquationProperty>(0));
      };
    }
  }

  namespace matching
  {
    template<
        typename VariableProperty,
        typename AccessProperty = internal::matching::EmptyAccessProperty>
    class Access
    {
      public:
        using Property = AccessProperty;

        Access(const VariableProperty& variable, AccessFunction accessFunction, AccessProperty property)
            : variable(VariableTraits<VariableProperty>::getId(&variable)),
              accessFunction(std::move(accessFunction)),
              property(std::move(property))
        {
        }

        template<typename... T>
        Access(VariableProperty variable, T&& ... accesses)
            : variable(VariableTraits<VariableProperty>::getId(&variable)),
              accessFunction(llvm::ArrayRef<DimensionAccess>({std::forward<T>(accesses)...}))
        {
        }

        typename VariableTraits<VariableProperty>::Id getVariable() const
        {
          return variable;
        }

        AccessFunction getAccessFunction() const
        {
          return accessFunction;
        }

        const AccessProperty& getProperty() const
        {
          return property;
        }

      private:
        typename VariableTraits<VariableProperty>::Id variable;
        AccessFunction accessFunction;
        AccessProperty property;
    };
  }

  namespace internal::matching
  {
    template<typename T>
    void insertOrAdd(std::map<T, IndexSet>& map, T key, IndexSet value)
    {
      if (auto it = map.find(key); it != map.end()) {
        it->second += std::move(value);
      } else {
        map.emplace(key, std::move(value));
      }
    }

    /// Graph node representing an equation.
    template<typename EquationProperty>
    class EquationVertex : public Matchable, public Dumpable
    {
      public:
        using Property = EquationProperty;
        using Traits = typename ::marco::modeling::matching::EquationTraits<EquationProperty>;
        using Id = typename Traits::Id;

        using Access = ::marco::modeling::matching::Access<
            typename Traits::VariableType,
            typename get_access_property<EquationProperty>::type>;

        EquationVertex(EquationProperty property)
            : Matchable(IndexSet(getIterationRanges(property))),
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
          os << tree_property << "Iteration ranges: " << getIterationRanges() << "\n";
          os << tree_property << "Matched: " << getMatched() << "\n";

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
          return Traits::getId(&property);
        }

        size_t getNumOfIterationVars() const
        {
          return getNumOfIterationVars(property);
        }

        MultidimensionalRange getIterationRanges() const
        {
          return getIterationRanges(property);
        }

        unsigned int flatSize() const
        {
          return getIterationRanges().flatSize();
        }

        std::vector<Access> getVariableAccesses() const
        {
          return Traits::getAccesses(&property);
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
          return Traits::getNumOfIterationVars(&p);
        }

        static MultidimensionalRange getIterationRanges(const EquationProperty& p)
        {
          return Traits::getIterationRanges(&p);
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
        using AccessProperty = typename Equation::Access::Property;

        Edge(typename Equation::Id equation,
             typename Variable::Id variable,
             MultidimensionalRange equationRanges,
             MultidimensionalRange variableRanges,
             typename Equation::Access access)
            : equation(std::move(equation)),
              variable(std::move(variable)),
              accessFunction(access.getAccessFunction()),
              accessProperty(access.getProperty()),
              incidenceMatrix(equationRanges, variableRanges),
              matchMatrix(equationRanges, variableRanges),
              visible(true)
        {
          incidenceMatrix.apply(getAccessFunction());
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

        const AccessFunction& getAccessFunction() const
        {
          return accessFunction;
        }

        const AccessProperty& getAccessProperty() const
        {
          return accessProperty;
        }

        const MCIM& getIncidenceMatrix() const
        {
          return incidenceMatrix;
        }

        void addMatch(const MCIM& match)
        {
          matchMatrix += match;
        }

        void removeMatch(const MCIM& match)
        {
          matchMatrix -= match;
        }

        const MCIM& getMatched() const
        {
          return matchMatrix;
        }

        MCIM getUnmatched() const
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

        AccessFunction accessFunction;
        AccessProperty accessProperty;
        MCIM incidenceMatrix;
        MCIM matchMatrix;

        // TODO: move to graph
        bool visible;
    };

    template<typename Graph>
    class BFSStep : public Dumpable
    {
      public:
        using VertexDescriptor = typename Graph::VertexDescriptor;
        using EdgeDescriptor = typename Graph::EdgeDescriptor;

        BFSStep(const Graph& graph,
                VertexDescriptor node,
                IndexSet candidates)
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
                IndexSet candidates,
                MCIM mappedFlow)
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

        ~BFSStep() = default;

        BFSStep& operator=(const BFSStep& other);

        template<typename G>
        friend void swap(BFSStep<G>& first, BFSStep<G>& second);

        using Dumpable::dump;

        void dump(std::ostream& stream) const override
        {
          using namespace marco::utils;

          TreeOStream os(stream);
          os << "BFS step\n";

          auto idVisitor = [](const auto& vertex) { return vertex.getId(); };

          os << tree_property << "Node: " << std::visit(idVisitor, (*graph)[getNode()]) << "\n";
          os << tree_property << "Candidates:\n" << getCandidates();

          if (hasPrevious()) {
            os << "\n";
            os << tree_property << "Edge: "
               << std::visit(idVisitor, (*graph)[getEdge().from])
               << " - "
               << std::visit(idVisitor, (*graph)[getEdge().to])
               << "\n";

            os << tree_property << "Mapped flow:\n" << getMappedFlow() << "\n";

            os << tree_property << "Previous:\n";
            getPrevious()->dump(os);
          }
        }

        bool hasPrevious() const
        {
          return previous.get() != nullptr;
        }

        const BFSStep *getPrevious() const
        {
          return previous.get();
        }

        const VertexDescriptor& getNode() const
        {
          return node;
        }

        const IndexSet& getCandidates() const
        {
          return candidates;
        }

        const EdgeDescriptor& getEdge() const
        {
          assert(edge.hasValue());
          return *edge;
        }

        const MCIM& getMappedFlow() const
        {
          assert(mappedFlow.hasValue());
          return *mappedFlow;
        }

      private:
        // Stored for debugging purpose
        const Graph* graph;

        std::unique_ptr<BFSStep> previous;
        VertexDescriptor node;
        IndexSet candidates;
        llvm::Optional<EdgeDescriptor> edge;
        llvm::Optional<MCIM> mappedFlow;
    };

    template<typename Graph>
    void swap(BFSStep<Graph>& first, BFSStep<Graph>& second)
    {
      using std::swap;

      swap(first.previous, second.previous);
      swap(first.node, second.node);
      swap(first.candidates, second.candidates);
      swap(first.edge, second.edge);
      swap(first.mappedFlow, second.mappedFlow);
    }

    template<typename Graph>
    BFSStep<Graph>& BFSStep<Graph>::operator=(const BFSStep<Graph>& other)
    {
      BFSStep<Graph> result(other);
      swap(*this, result);
      return *this;
    }

    template<typename BFSStep>
    class Frontier : public Dumpable
    {
      private:
        template<typename T> using Container = std::vector<T>;

      public:
        using iterator = typename Container<BFSStep>::iterator;
        using const_iterator = typename Container<BFSStep>::const_iterator;

        using Dumpable::dump;

        void dump(std::ostream& stream) const override
        {
          using namespace marco::utils;

          TreeOStream os(stream);
          os << "Frontier\n";

          for (const auto& step : steps) {
            os << tree_property;
            step.dump(os);
            os << "\n";
          }
        }

        friend void swap(Frontier& first, Frontier& second)
        {
          using std::swap;

          swap(first.steps, second.steps);
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
        void emplace(Args&& ... args)
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

        iterator begin()
        {
          return steps.begin();
        }

        const_iterator begin() const
        {
          return steps.begin();
        }

        iterator end()
        {
          return steps.end();
        }

        const_iterator end() const
        {
          return steps.end();
        }

      private:
        Container<BFSStep> steps;
    };

    template<typename Graph>
    class Flow : public Dumpable
    {
      private:
        using VertexDescriptor = typename Graph::VertexDescriptor;
        using EdgeDescriptor = typename Graph::EdgeDescriptor;

      public:
        Flow(const Graph& graph, VertexDescriptor source, EdgeDescriptor edge, const MCIM& delta)
            : graph(&graph),
              source(std::move(source)),
              edge(std::move(edge)),
              delta(std::move(delta))
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

      private:
        // Stored for debugging purpose
        const Graph* graph;

      public:
        const VertexDescriptor source;
        const EdgeDescriptor edge;
        const MCIM delta;
    };

    template<typename Flow>
    class AugmentingPath : public Dumpable
    {
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

          for (const auto& flow : flows) {
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

    /// Represents how an equation has been matched (i.e. the selected indexes and access).
    template<typename EquationProperty, typename VariableProperty, typename AccessProperty>
    class MatchingSolution
    {
      public:
        MatchingSolution(
            EquationProperty equation,
            VariableProperty variable,
            MultidimensionalRange indexes,
            AccessProperty access)
            : equation(std::move(equation)),
              variable(std::move(variable)),
              indexes(std::move(indexes)),
              access(std::move(access))
        {
        }

        EquationProperty& getEquation()
        {
          return equation;
        }

        const EquationProperty& getEquation() const
        {
          return equation;
        }

        const VariableProperty& getVariable() const
        {
          return variable;
        }

        const AccessProperty& getAccess() const
        {
          return access;
        }

        const MultidimensionalRange& getIndexes() const
        {
          return indexes;
        }

      private:
        EquationProperty equation;
        VariableProperty variable;
        MultidimensionalRange indexes;
        AccessProperty access;
    };
  }

  template<typename VariableProperty, typename EquationProperty>
  class MatchingGraph : public internal::Dumpable
  {
    public:
      using Variable = internal::matching::VariableVertex<VariableProperty>;
      using Equation = internal::matching::EquationVertex<EquationProperty>;
      using Vertex = std::variant<Variable, Equation>;
      using Edge = internal::matching::Edge<Variable, Equation>;

    private:
      using Graph = internal::UndirectedGraph<Vertex, Edge>;

      using VertexDescriptor = typename Graph::VertexDescriptor;
      using EdgeDescriptor = typename Graph::EdgeDescriptor;

      using VertexIterator = typename Graph::VertexIterator;
      using EdgeIterator = typename Graph::EdgeIterator;
      using VisibleIncidentEdgeIterator = typename Graph::FilteredIncidentEdgeIterator;

      using MCIM = internal::MCIM;
      using BFSStep = internal::matching::BFSStep<Graph>;
      using Frontier = internal::matching::Frontier<BFSStep>;
      using Flow = internal::matching::Flow<Graph>;
      using AugmentingPath = internal::matching::AugmentingPath<Flow>;

    public:
      using VariableIterator = typename Graph::FilteredVertexIterator;
      using EquationIterator = typename Graph::FilteredVertexIterator;

      using AccessProperty = typename Equation::Access::Property;
      using Access = matching::Access<VariableProperty, AccessProperty>;
      using MatchingSolution = internal::matching::MatchingSolution<EquationProperty, VariableProperty, AccessProperty>;

      using Dumpable::dump;

      void dump(std::ostream& stream) const override
      {
        using namespace marco::utils;

        TreeOStream os(stream);
        os << "Matching graph\n";

        for (auto descriptor : graph.getVertices()) {
          std::visit(
              [&](const auto& vertex) {
                os << tree_property;
                vertex.dump(os);
              }, graph[descriptor]);
        }

        for (auto descriptor : graph.getEdges()) {
          os << tree_property;
          graph[descriptor].dump(os);
        }
      }

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

        auto accesses = equation.getVariableAccesses();

        // The equation may access multiple variables or even multiple indexes
        // of the same variable. Add an edge to the graph for each of those
        // accesses.

        for (const auto& access : accesses) {
          auto variableDescriptor = getVariableVertex(access.getVariable());
          Variable& variable = getVariable(variableDescriptor);

          auto indices = variable.getIndices();
          for (const auto& range : indices.getRanges()) {
            Edge edge(equation.getId(), variable.getId(), equation.getIterationRanges(), range, access);
            graph.addEdge(equationDescriptor, variableDescriptor, edge);
          }
        }
      }

      /// Get the total amount of scalar variables inside the graph.
      size_t getNumberOfScalarVariables() const
      {
        size_t result = 0;

        for (auto variableDescriptor : getVariables()) {
          result += getVariable(variableDescriptor).flatSize();
        }

        return result;
      }

      /// Get the total amount of scalar equations inside the graph.
      /// With "scalar equations" we mean the ones generated by unrolling
      /// the loops defining them.
      size_t getNumberOfScalarEquations() const
      {
        size_t result = 0;

        for (auto equationDescriptor : getEquations()) {
          result += getEquation(equationDescriptor).flatSize();
        }

        return result;
      }

      bool allNodesMatched() const
      {
        auto vertices = graph.getVertices();

        auto allComponentsMatchedFn = [](const auto& obj) {
          return obj.allComponentsMatched();
        };

        return llvm::all_of(
            vertices, [&](const auto& descriptor) {
              return std::visit(allComponentsMatchedFn, graph[descriptor]);
            });
      }

      bool hasEdge(typename Equation::Id equationId, typename Variable::Id variableId) const
      {
        if (findEdge<Equation, Variable>(equationId, variableId).first) {
          return true;
        }

        return findEdge<Variable, Equation>(variableId, equationId).first;
      }

      EdgeDescriptor getFirstOutVisibleEdge(VertexDescriptor vertex) const
      {
        auto edges = getVisibleEdges(vertex);
        assert(edges.begin() != edges.end() && "Vertex doesn't belong to any edge");
        return *edges.begin();
      }

      /// Apply the simplification algorithm in order to perform all
      /// the obligatory matches, that is the variables and equations
      /// having only one incident edge.
      ///
      /// @return true if the simplification algorithm didn't find any inconsistency
      bool simplify()
      {
        // Vertices that are candidate for the first simplification phase.
        // They are the ones having only one incident edge.
        std::list<VertexDescriptor> candidates;

        // Determine the initial set of vertices with exactly one incident edge
        for (VertexDescriptor vertex : graph.getVertices()) {
          auto incidentEdges = getVertexVisibilityDegree(vertex);

          if (incidentEdges == 0) {
            return false;
          }

          if (incidentEdges == 1) {
            candidates.push_back(vertex);
          }
        }

        // Iterate on the candidate vertices and apply the simplification algorithm
        while (!candidates.empty()) {
          VertexDescriptor v1 = candidates.front();
          candidates.pop_front();

          auto edgeDescriptor = getFirstOutVisibleEdge(v1);
          Edge& edge = graph[edgeDescriptor];

          VertexDescriptor v2 = edgeDescriptor.from == v1 ? edgeDescriptor.to : edgeDescriptor.from;

          const auto& u = edge.getIncidenceMatrix();

          assert(u.getEquationRanges().isSingleMultidimensionalRange());
          assert(u.getVariableRanges().isSingleMultidimensionalRange());

          auto matchOptions = internal::solveLocalMatchingProblem(
              u.getEquationRanges().minContainingRange(),//todo remove .minContainingRange()
              u.getVariableRanges().minContainingRange(),//todo remove .minContainingRange()
              edge.getAccessFunction());

          // The simplification steps is executed only in case of a single
          // matching option. In case of multiple ones, in fact, the choice
          // would be arbitrary and may affect the feasibility of the
          // array-aware matching problem.

          if (matchOptions.size() == 1) {
            const MCIM& match = matchOptions[0];

            Variable& variable = isVariable(v1) ? getVariable(v1) : getVariable(v2);
            Equation& equation = isEquation(v1) ? getEquation(v1) : getEquation(v2);

            auto proposedVariableMatch = match.flattenRows();
            auto proposedEquationMatch = match.flattenColumns();

            MCIM reducedMatch = match.filterColumns(proposedVariableMatch - variable.getMatched());
            reducedMatch = reducedMatch.filterRows(proposedEquationMatch - equation.getMatched());

            edge.addMatch(reducedMatch);

            auto reducedVariableMatch = reducedMatch.flattenRows();
            auto reducedEquationMatch = reducedMatch.flattenColumns();

            variable.addMatch(reducedVariableMatch);
            equation.addMatch(reducedEquationMatch);

            auto allComponentsMatchedVisitor = [](const auto& vertex) -> bool {
              return vertex.allComponentsMatched();
            };

            if (!std::visit(allComponentsMatchedVisitor, graph[v1])) {
              return false;
            }

            bool shouldRemoveOppositeNode = std::visit(allComponentsMatchedVisitor, graph[v2]);

            // Remove the edge and the current candidate vertex
            remove(edgeDescriptor);
            remove(v1);

            if (shouldRemoveOppositeNode) {
              // When a node is removed, then also its incident edges are
              // removed. This can lead to new obliged matches, like in the
              // following example:
              //        |-- v3 ----
              // v1 -- v2         |
              //        |-- v4 -- v5
              // v1 is the current candidate and thus is removed.
              // v2 is removed because fully matched.
              // v3 and v4 become new candidates for the simplification pass.

              for (auto e : graph.getOutgoingEdges(v2)) {
                remove(e);

                VertexDescriptor v = e.from == v2 ? e.to : e.from;

                // TODO: remove visibility filter once the visibility is moved to the base graph
                auto isOtherVertexVisibleFn = [](const auto& obj) -> bool {
                  return obj.isVisible();
                };

                if (!std::visit(isOtherVertexVisibleFn, graph[v])) {
                  continue;
                }

                auto visibilityDegree = getVertexVisibilityDegree(v);

                if (visibilityDegree == 0) {
                  // Chained simplifications may have led the 'v' vertex
                  // without any edge. In that case, it must have been fully
                  // matched during the process.

                  if (!std::visit(allComponentsMatchedVisitor, graph[v])) {
                    return false;
                  }

                  // 'v' will also be present for sure in the candidates list.
                  // However, being fully matched and having no outgoing edge,
                  // we now must remove it.
                  candidates.remove_if([&](auto vertex) {
                    return vertex == v;
                  });
                } else if (visibilityDegree == 1) {
                  candidates.push_back(v);
                }
              }

              // Remove the v2 vertex and remove it from the candidates
              remove(v2);

              candidates.remove_if([&](auto v) {
                return v == v2;
              });
            } else {
              // When an edge is removed but one of its vertices survives, we must
              // check if the remaining vertex has an obliged match.

              auto visibilityDegree = getVertexVisibilityDegree(v2);

              if (visibilityDegree == 1) {
                candidates.push_back(v2);
              }
            }
          }
        }

        return true;
      }

      /// Apply the matching algorithm.
      ///
      /// @return true if the matching algorithm managed to fully match all the nodes
      bool match()
      {
        if (allNodesMatched()) {
          return true;
        }

        bool success;
        bool complete;

        do {
          success = matchIteration();
          complete = allNodesMatched();
        } while (success && !complete);

        return success;
      }

      /// Get the solution of the matching problem.
      std::vector<MatchingSolution> getMatch() const
      {
        assert(allNodesMatched() && "Not all the nodes have been fully matched");
        std::vector<MatchingSolution> result;

        for (auto equationDescriptor : getEquations()) {
          for (auto edgeDescriptor : getEdges(equationDescriptor)) {
            const Edge& edge = graph[edgeDescriptor];

            if (const auto& matched = edge.getMatched(); !matched.empty()) {
              auto variableDescriptor =
                  edgeDescriptor.from == equationDescriptor ? edgeDescriptor.to : edgeDescriptor.from;

              IndexSet allIndexes = matched.flattenColumns();

              for (const auto& groupedIndexes : allIndexes.getRanges()) {
                result.emplace_back(
                    getEquation(equationDescriptor).getProperty(),
                    getVariable(variableDescriptor).getProperty(),
                    groupedIndexes,
                    edge.getAccessProperty());
              }
            }
          }
        }

        return result;
      }

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

        auto it = std::find_if(
            vertices.begin(), vertices.end(), [&](const VertexDescriptor& v) {
              const auto& vertex = graph[v];

              if (!std::holds_alternative<T>(vertex)) {
                return false;
              }

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
        std::visit(
            [](auto& obj) -> void {
              obj.setVisibility(false);
            }, graph[vertex]);
      }

      template<typename From, typename To>
      std::pair<bool, EdgeIterator> findEdge(typename From::Id from, typename To::Id to) const
      {
        auto edges = graph.getEdges();

        auto it = std::find_if(
            edges.begin(), edges.end(), [&](const EdgeDescriptor& e) {
              auto& source = graph[e.from];
              auto& target = graph[e.to];

              if (!std::holds_alternative<From>(source) || !std::holds_alternative<To>(target)) {
                return false;
              }

              return std::get<From>(source).getId() == from && std::get<To>(target).getId() == to;
            });

        return std::make_pair(it != edges.end(), it);
      }

      auto getEdges(VertexDescriptor vertex) const
      {
        return graph.getOutgoingEdges(vertex);
      }

      llvm::iterator_range<VisibleIncidentEdgeIterator> getVisibleEdges(VertexDescriptor vertex) const
      {
        auto filter = [&](const Edge& edge) -> bool {
          return edge.isVisible();
        };

        return graph.getOutgoingEdges(vertex, filter);
      }

      void remove(EdgeDescriptor edge)
      {
        graph[edge].setVisibility(false);
      }

      bool matchIteration()
      {
        llvm::SmallVector<AugmentingPath, 8> augmentingPaths;
        getAugmentingPaths(augmentingPaths);

        if (augmentingPaths.empty()) {
          return false;
        }

        for (auto& path : augmentingPaths) {
          applyPath(path);
        }

        return true;
      }

      void getAugmentingPaths(llvm::SmallVectorImpl<AugmentingPath>& paths) const
      {
        auto sortHeuristic = [](const BFSStep& first, const BFSStep& second) {
          return first.getCandidates().flatSize() > second.getCandidates().flatSize();
        };

        Frontier frontier;

        // Calculation of the initial frontier
        auto equations = getEquations();

        for (auto equationDescriptor : equations) {
          const Equation& equation = getEquation(equationDescriptor);

          if (auto unmatchedEquations = equation.getUnmatched(); !unmatchedEquations.empty()) {
            frontier.emplace(BFSStep(graph, equationDescriptor, std::move(unmatchedEquations)));
          }
        }

        std::sort(frontier.begin(), frontier.end(), sortHeuristic);

        // Breadth-first search
        Frontier newFrontier;
        Frontier foundPaths;

        while (!frontier.empty() && foundPaths.empty()) {
          for (const BFSStep& step : frontier) {
            auto vertexDescriptor = step.getNode();

            for (EdgeDescriptor edgeDescriptor : getEdges(vertexDescriptor)) {
              assert(edgeDescriptor.from == vertexDescriptor);
              VertexDescriptor nextNode = edgeDescriptor.to;
              const Edge& edge = graph[edgeDescriptor];

              if (isEquation(vertexDescriptor)) {
                assert(isVariable(nextNode));
                auto unmatchedMatrix = edge.getUnmatched();
                auto filteredMatrix = unmatchedMatrix.filterRows(step.getCandidates());
                internal::LocalMatchingSolutions solutions = internal::solveLocalMatchingProblem(filteredMatrix);

                for (auto solution : solutions) {
                  Variable var = getVariable(edgeDescriptor.to);
                  auto unmatchedScalarVariables = var.getUnmatched();
                  auto matched = solution.filterColumns(unmatchedScalarVariables);

                  if (!matched.empty()) {
                    foundPaths.emplace(graph, step, edgeDescriptor, nextNode, matched.flattenRows(), matched);
                  } else {
                    newFrontier.emplace(graph, step, edgeDescriptor, nextNode, solution.flattenRows(), solution);
                  }
                }
              } else {
                assert(isEquation(nextNode));
                auto filteredMatrix = edge.getMatched().filterColumns(step.getCandidates());
                internal::LocalMatchingSolutions solutions = internal::solveLocalMatchingProblem(filteredMatrix);

                for (auto solution : solutions) {
                  newFrontier.emplace(graph, step, edgeDescriptor, nextNode, solution.flattenColumns(), solution);
                }
              }
            }
          }

          // Set the new frontier for the next iteration
          frontier.clear();
          frontier.swap(newFrontier);

          std::sort(frontier.begin(), frontier.end(), sortHeuristic);
        }

        std::sort(foundPaths.begin(), foundPaths.end(), sortHeuristic);

        // For each traversed node, keep track of the indexes that have already
        // been traversed by some augmenting path. A new candidate path can be
        // accepted only if it does not traverse any of them.
        std::map<VertexDescriptor, IndexSet> visited;

        for (const BFSStep& pathEnd : foundPaths) {
          // All the candidate paths consist in at least two nodes by construction
          assert(pathEnd.hasPrevious());

          std::list<Flow> flows;

          // The path's validity is unknown, so we must avoid polluting the
          // list of visited scalar nodes. If the path will be marked as valid,
          // then the new visits will be merged with the already found ones.
          std::map<VertexDescriptor, IndexSet> newVisits;

          const BFSStep* curStep = &pathEnd;
          MCIM map = curStep->getMappedFlow();
          bool validPath = true;

          while (curStep && validPath) {
            if (curStep->hasPrevious()) {
              if (!flows.empty()) {
                // Restrict the flow
                const auto& prevMap = flows.front().delta;

                if (isVariable(curStep->getNode())) {
                  map = curStep->getMappedFlow().filterColumns(prevMap.flattenRows());
                } else {
                  map = curStep->getMappedFlow().filterRows(prevMap.flattenColumns());
                }
              }

              flows.emplace(flows.begin(), graph, curStep->getPrevious()->getNode(), curStep->getEdge(), map);
            }

            auto touchedIndexes = isVariable(curStep->getNode()) ? map.flattenRows() : map.flattenColumns();

            if (auto it = visited.find(curStep->getNode()); it != visited.end()) {
              auto& alreadyTouchedIndices = it->second;

              if (touchedIndexes.overlaps(alreadyTouchedIndices)) {
                // The current path intersects another one, so we need to discard it
                validPath = false;
              } else {
                insertOrAdd(newVisits, curStep->getNode(), alreadyTouchedIndices + touchedIndexes);
              }
            } else {
              insertOrAdd(newVisits, curStep->getNode(), touchedIndexes);
            }

            // Move backwards inside the candidate augmenting path
            curStep = curStep->getPrevious();
          }

          if (validPath) {
            paths.emplace_back(std::move(flows));

            for (auto& p : newVisits) {
              visited.insert_or_assign(p.first, p.second);
            }
          }
        }
      }

      /// Apply an augmenting path to the graph.
      void applyPath(const AugmentingPath& path)
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

        std::map<VertexDescriptor, IndexSet> removedMatches;
        std::map<VertexDescriptor, IndexSet> newMatches;

        // Update the match matrices on the edges and store the information
        // about the vertices to be updated later.

        for (auto& flow : path) {
          auto& edge = graph[flow.edge];

          VertexDescriptor from = flow.source;
          VertexDescriptor to = flow.edge.from == from ? flow.edge.to : flow.edge.from;

          auto deltaEquations = flow.delta.flattenColumns();
          auto deltaVariables = flow.delta.flattenRows();

          if (isVariable(from)) {
            // Backward node
            insertOrAdd(removedMatches, from, deltaVariables);
            insertOrAdd(removedMatches, to, deltaEquations);
            edge.removeMatch(flow.delta);
          } else {
            // Forward node
            insertOrAdd(newMatches, from, deltaEquations);
            insertOrAdd(newMatches, to, deltaVariables);
            edge.addMatch(flow.delta);
          }
        }

        // Update the match information stored on the vertices

        for (const auto& match : removedMatches) {
          std::visit(
              [&match](auto& node) {
                node.removeMatch(match.second);
              }, graph[match.first]);
        }

        for (const auto& match : newMatches) {
          std::visit(
              [&match](auto& node) {
                node.addMatch(match.second);
              }, graph[match.first]);
        }
      }

      Graph graph;
  };
}

#endif  // MARCO_MODELING_MATCHING_H
