#ifndef MARCO_CODEGEN_LOWERING_CLASSDEPENDENCYGRAPH_H
#define MARCO_CODEGEN_LOWERING_CLASSDEPENDENCYGRAPH_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/ClassPath.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include <set>

namespace marco::codegen::lowering {
class ClassDependencyGraph {
public:
  /// A node of the graph.
  struct Node {
    Node();

    Node(const ClassDependencyGraph *graph, ClassPath path);

    bool operator==(const Node &other) const;

    bool operator!=(const Node &other) const;

    bool operator<(const Node &other) const;

    const ClassDependencyGraph *graph;
    ClassPath path;
  };

  ClassDependencyGraph();

  virtual ~ClassDependencyGraph();

  /// Add a class to the graph.
  void addClass(ClassPath path) {
    Node node = nodes.emplace_back(this, std::move(path));
    nodesByPath[node.path] = node;
  }

  /// Discover the dependencies of the variables that have been added to
  /// the graph.
  void discoverDependencies() {
    for (const Node &node : nodes) {
      // Connect the entry node to the current one.
      arcs[getEntryNode().path].insert(node);

      // Connect the current node to the other nodes to which it depends.
      const ClassPath &path = node.path;
      auto &children = arcs[path];

      for (const ClassPath &dependency : getDependencies(path)) {
        children.insert(nodesByPath[dependency]);
      }
    }
  }

  /// Get the number of nodes.
  size_t getNumOfNodes() const;

  /// Get the entry node of the graph.
  const Node &getEntryNode() const;

  /// @name Iterators for the children of a node.
  /// {

  std::set<Node>::const_iterator childrenBegin(Node node) const;

  std::set<Node>::const_iterator childrenEnd(Node node) const;

  /// }
  /// @name Iterators for the nodes.
  /// {

  llvm::SmallVector<Node>::const_iterator nodesBegin() const;

  llvm::SmallVector<Node>::const_iterator nodesEnd() const;

  /// }

  /// Get the cycles of the graph.
  llvm::SmallVector<llvm::DenseSet<ClassPath>> getCycles() const;

  /// Perform a post-order visit of the nodes of the graph.
  llvm::SmallVector<ClassPath> postOrder() const;

  /// Perform a reverse post-order visit of the nodes of the graph.
  llvm::SmallVector<ClassPath> reversePostOrder() const;

protected:
  virtual std::set<ClassPath> getDependencies(const ClassPath &classPath) = 0;

private:
  llvm::DenseMap<ClassPath, std::set<Node>> arcs;
  llvm::SmallVector<Node> nodes;
  llvm::DenseMap<ClassPath, Node> nodesByPath;
};
} // namespace marco::codegen::lowering

namespace llvm {
template <>
struct DenseMapInfo<marco::codegen::lowering::ClassDependencyGraph::Node> {
  static inline marco::codegen::lowering::ClassDependencyGraph::Node
  getEmptyKey() {
    return {nullptr, {}};
  }

  static inline marco::codegen::lowering::ClassDependencyGraph::Node
  getTombstoneKey() {
    return {nullptr, {}};
  }

  static unsigned getHashValue(
      const marco::codegen::lowering::ClassDependencyGraph::Node &val) {
    return llvm::hash_value(val.path.get());
  }

  static bool
  isEqual(const marco::codegen::lowering::ClassDependencyGraph::Node &lhs,
          const marco::codegen::lowering::ClassDependencyGraph::Node &rhs) {
    return lhs.graph == rhs.graph && lhs.path == rhs.path;
  }
};

template <>
struct GraphTraits<const marco::codegen::lowering::ClassDependencyGraph *> {
  using GraphType = const marco::codegen::lowering::ClassDependencyGraph *;
  using NodeRef = marco::codegen::lowering::ClassDependencyGraph::Node;

  using ChildIteratorType = std::set<
      marco::codegen::lowering::ClassDependencyGraph::Node>::const_iterator;

  static NodeRef getEntryNode(const GraphType &graph) {
    return graph->getEntryNode();
  }

  static ChildIteratorType child_begin(NodeRef node) {
    return node.graph->childrenBegin(node);
  }

  static ChildIteratorType child_end(NodeRef node) {
    return node.graph->childrenEnd(node);
  }

  using nodes_iterator = llvm::SmallVector<NodeRef>::const_iterator;

  static nodes_iterator nodes_begin(GraphType *graph) {
    return (*graph)->nodesBegin();
  }

  static nodes_iterator nodes_end(GraphType *graph) {
    return (*graph)->nodesEnd();
  }

  // There is no need for a dedicated class for the arcs.
  using EdgeRef = marco::codegen::lowering::ClassDependencyGraph::Node;

  using ChildEdgeIteratorType = std::set<
      marco::codegen::lowering::ClassDependencyGraph::Node>::const_iterator;

  static ChildEdgeIteratorType child_edge_begin(NodeRef node) {
    return node.graph->childrenBegin(node);
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef node) {
    return node.graph->childrenEnd(node);
  }

  static NodeRef edge_dest(EdgeRef edge) { return edge; }

  static unsigned int size(GraphType *graph) {
    return (*graph)->getNumOfNodes();
  }
};
} // namespace llvm

#endif // MARCO_CODEGEN_LOWERING_CLASSDEPENDENCYGRAPH_H
