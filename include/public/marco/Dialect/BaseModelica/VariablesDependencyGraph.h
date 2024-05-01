#ifndef MARCO_DIALECT_BASEMODELICA_VARIABLESDEPENDENCYGRAPH_H
#define MARCO_DIALECT_BASEMODELICA_VARIABLESDEPENDENCYGRAPH_H

#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
#include <set>

namespace mlir::bmodelica
{
  class VariablesDependencyGraph
  {
    public:
      /// A node of the graph.
      struct Node
      {
        Node();

        Node(const VariablesDependencyGraph* graph, mlir::Operation* variable);

        bool operator==(const Node& other) const;

        bool operator!=(const Node& other) const;

        bool operator<(const Node& other) const;

        const VariablesDependencyGraph* graph;
        mlir::Operation* variable;
      };

      VariablesDependencyGraph();

      virtual ~VariablesDependencyGraph();

      /// Add a group of variables to the graph and optionally enforce their
      /// relative order to be preserved.
      void addVariables(llvm::ArrayRef<VariableOp> variables);

      /// Discover the dependencies of the variables that have been added to
      /// the graph.
      void discoverDependencies();

      /// Get the number of nodes.
      /// Note that nodes consists in the inserted variables together with the
      /// entry node.
      [[nodiscard]] size_t getNumOfNodes() const;

      /// Get the entry node of the graph.
      [[nodiscard]] const Node& getEntryNode() const;

      /// @name Iterators for the children of a node
      /// {

      [[nodiscard]] std::set<Node>::const_iterator
      childrenBegin(Node node) const;

      [[nodiscard]] std::set<Node>::const_iterator
      childrenEnd(Node node) const;

      /// }
      /// @name Iterators for the nodes
      /// {

      [[nodiscard]] llvm::SmallVector<Node>::const_iterator nodesBegin() const;

      [[nodiscard]] llvm::SmallVector<Node>::const_iterator nodesEnd() const;

      /// }

      /// Check if the graph contains cycles.
      bool hasCycles() const;

      /// Perform a post-order visit of the graph and get the ordered
      /// variables.
      llvm::SmallVector<VariableOp> postOrder() const;

      /// Perform a reverse post-order visit of the graph and get the ordered
      /// variables.
      llvm::SmallVector<VariableOp> reversePostOrder() const;

    protected:
      // TODO replace llvm::StringRef with mlir::StringAttr for safety.
      virtual std::set<llvm::StringRef> getDependencies(
          VariableOp variable) = 0;

    private:
      llvm::DenseMap<mlir::Operation*, std::set<Node>> arcs;
      llvm::SmallVector<Node> nodes;
      llvm::StringMap<Node> nodesByName;
  };
}

namespace llvm
{
  template<>
  struct DenseMapInfo<::mlir::bmodelica::VariablesDependencyGraph::Node>
  {
    static inline ::mlir::bmodelica::VariablesDependencyGraph::Node
    getEmptyKey()
    {
      return {nullptr, nullptr};
    }

    static inline ::mlir::bmodelica::VariablesDependencyGraph::Node
    getTombstoneKey()
    {
      return {nullptr, nullptr};
    }

    static unsigned getHashValue(
        const ::mlir::bmodelica::VariablesDependencyGraph::Node& val)
    {
      return std::hash<mlir::Operation*>{}(val.variable);
    }

    static bool isEqual(
        const ::mlir::bmodelica::VariablesDependencyGraph::Node& lhs,
        const ::mlir::bmodelica::VariablesDependencyGraph::Node& rhs)
    {
      return lhs.graph == rhs.graph && lhs.variable == rhs.variable;
    }
  };

  template<>
  struct GraphTraits<const ::mlir::bmodelica::VariablesDependencyGraph*>
  {
    using GraphType = const ::mlir::bmodelica::VariablesDependencyGraph*;
    using NodeRef = ::mlir::bmodelica::VariablesDependencyGraph::Node;

    using ChildIteratorType =
        std::set<::mlir::bmodelica::VariablesDependencyGraph::Node>
            ::const_iterator;

    static NodeRef getEntryNode(const GraphType& graph)
    {
      return graph->getEntryNode();
    }

    static ChildIteratorType child_begin(NodeRef node)
    {
      return node.graph->childrenBegin(node);
    }

    static ChildIteratorType child_end(NodeRef node)
    {
      return node.graph->childrenEnd(node);
    }

    using nodes_iterator = llvm::SmallVector<NodeRef>::const_iterator;

    static nodes_iterator nodes_begin(GraphType* graph)
    {
      return (*graph)->nodesBegin();
    }

    static nodes_iterator nodes_end(GraphType* graph)
    {
      return (*graph)->nodesEnd();
    }

    // There is no need for a dedicated class for the arcs.
    using EdgeRef = ::mlir::bmodelica::VariablesDependencyGraph::Node;

    using ChildEdgeIteratorType =
        std::set<::mlir::bmodelica::VariablesDependencyGraph::Node>::const_iterator;

    static ChildEdgeIteratorType child_edge_begin(NodeRef node)
    {
      return node.graph->childrenBegin(node);
    }

    static ChildEdgeIteratorType child_edge_end(NodeRef node)
    {
      return node.graph->childrenEnd(node);
    }

    static NodeRef edge_dest(EdgeRef edge)
    {
      return edge;
    }

    static unsigned int size(GraphType* graph)
    {
      return (*graph)->getNumOfNodes();
    }
  };
}

#endif // MARCO_DIALECT_BASEMODELICA_VARIABLESDEPENDENCYGRAPH_H
