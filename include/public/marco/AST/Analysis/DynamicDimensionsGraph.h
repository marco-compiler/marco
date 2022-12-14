#ifndef MARCO_AST_ANALYSIS_DYNAMICDIMENSIONSGRAPH_H
#define MARCO_AST_ANALYSIS_DYNAMICDIMENSIONSGRAPH_H

#include "marco/AST/AST.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include <set>

using namespace ::marco::ast;

namespace marco::ast
{
  /// Directed graph representing the dependencies among the variables with
  /// respect to the usage of variables for the computation of the dynamic
  /// dimensions of their types.
  class DynamicDimensionsGraph
  {
    public:
      /// A node of the graph.
      struct Node
      {
        Node();

        Node(const DynamicDimensionsGraph* graph, const Member* member);

        bool operator==(const Node& other) const;

        bool operator!=(const Node& other) const;

        bool operator<(const Node& other) const;

        const DynamicDimensionsGraph* graph;
        const Member* member;
      };

      DynamicDimensionsGraph();

      void addMembersGroup(llvm::ArrayRef<const Member*> group);

      void discoverDependencies();

      /// Get the number of nodes.
      /// Note that nodes consists in the inserted members together with the
      /// entry node.
      size_t getNumOfNodes() const;

      Node getEntryNode() const;

      /// @name Iterators for the children of a node
      /// {

      std::set<Node>::const_iterator childrenBegin(Node node) const;

      std::set<Node>::const_iterator childrenEnd(Node node) const;

      /// }
      /// @name Iterators for the nodes
      /// {

      llvm::SmallVector<Node>::const_iterator nodesBegin() const;

      llvm::SmallVector<Node>::const_iterator nodesEnd() const;

      /// }

      /// Check if the graph contains cycles.
      bool hasCycles() const;

      /// Perform a post-order visit of the graph and get the ordered members.
      llvm::SmallVector<const Member*> postOrder() const;

    private:
      std::set<const Member*> getUsedMembers(const Expression* expression);

    private:
      llvm::DenseMap<const Member*, std::set<Node>> arcs;
      llvm::SmallVector<Node> nodes;
      llvm::DenseMap<llvm::StringRef, Node> symbolTable;
  };
}

namespace llvm
{
  template<>
  struct DenseMapInfo<DynamicDimensionsGraph::Node>
  {
    static inline DynamicDimensionsGraph::Node getEmptyKey()
    {
      return DynamicDimensionsGraph::Node(nullptr, nullptr);
    }

    static inline DynamicDimensionsGraph::Node getTombstoneKey()
    {
      return DynamicDimensionsGraph::Node(nullptr, nullptr);
    }

    static unsigned getHashValue(const DynamicDimensionsGraph::Node& val)
    {
      return std::hash<const Member*>{}(val.member);
    }

    static bool isEqual(
        const DynamicDimensionsGraph::Node& lhs,
        const DynamicDimensionsGraph::Node& rhs)
    {
      return lhs.graph == rhs.graph && lhs.member == rhs.member;
    }
  };

  template<>
  struct GraphTraits<const DynamicDimensionsGraph*>
  {
    using GraphType = const DynamicDimensionsGraph*;
    using NodeRef = DynamicDimensionsGraph::Node;

    using ChildIteratorType =
        std::set<DynamicDimensionsGraph::Node>::const_iterator;

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
    using EdgeRef = DynamicDimensionsGraph::Node;

    using ChildEdgeIteratorType =
        std::set<DynamicDimensionsGraph::Node>::const_iterator;

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

#endif // MARCO_AST_ANALYSIS_DYNAMICDIMENSIONSGRAPH_H
