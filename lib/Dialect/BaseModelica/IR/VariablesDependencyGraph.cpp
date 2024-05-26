#include "marco/Dialect/BaseModelica/IR/VariablesDependencyGraph.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica
{
  VariablesDependencyGraph::Node::Node()
      : graph(nullptr), variable(nullptr)
  {
  }

  VariablesDependencyGraph::Node::Node(
      const VariablesDependencyGraph* graph, mlir::Operation* variable)
      : graph(graph), variable(variable)
  {
  }

  bool VariablesDependencyGraph::Node::operator==(const Node& other) const
  {
    return graph == other.graph && variable == other.variable;
  }

  bool VariablesDependencyGraph::Node::operator!=(const Node& other) const
  {
    return graph != other.graph || variable != other.variable;
  }

  bool VariablesDependencyGraph::Node::operator<(const Node& other) const
  {
    if (variable == nullptr) {
      return true;
    }

    if (other.variable == nullptr) {
      return false;
    }

    return variable < other.variable;
  }

  VariablesDependencyGraph::VariablesDependencyGraph()
  {
    // Entry node, which is connected to every other node.
    nodes.emplace_back(this, nullptr);

    // Ensure that the set of children for the entry node exists, even in
    // case of no other nodes.
    arcs[getEntryNode().variable] = {};
  }

  VariablesDependencyGraph::~VariablesDependencyGraph() = default;

  void VariablesDependencyGraph::addVariables(
      llvm::ArrayRef<VariableOp> variables)
  {
    for (VariableOp variable : variables) {
      assert(variable != nullptr);

      Node node(this, variable.getOperation());
      nodes.push_back(node);
      nodesByName[variable.getSymName()] = node;
    }
  }

  void VariablesDependencyGraph::discoverDependencies()
  {
    for (const Node& node : nodes) {
      if (node == getEntryNode()) {
        continue;
      }

      // Connect the entry node to the current one.
      arcs[getEntryNode().variable].insert(node);

      // Connect the current node to the other nodes to which it depends.
      mlir::Operation* variable = node.variable;
      auto& children = arcs[variable];

      auto variableOp = mlir::cast<VariableOp>(variable);

      for (llvm::StringRef dependency : getDependencies(variableOp)) {
        children.insert(nodesByName[dependency]);
      }
    }
  }

  size_t VariablesDependencyGraph::getNumOfNodes() const
  {
    return nodes.size();
  }

  const VariablesDependencyGraph::Node&
  VariablesDependencyGraph::getEntryNode() const
  {
    return nodes[0];
  }

  std::set<VariablesDependencyGraph::Node>::const_iterator
  VariablesDependencyGraph::childrenBegin(Node node) const
  {
    auto it = arcs.find(node.variable);
    assert(it != arcs.end());
    const auto& children = it->second;
    return children.begin();
  }

  std::set<VariablesDependencyGraph::Node>::const_iterator
  VariablesDependencyGraph::childrenEnd(Node node) const
  {
    auto it = arcs.find(node.variable);
    assert(it != arcs.end());
    const auto& children = it->second;
    return children.end();
  }

  llvm::SmallVector<VariablesDependencyGraph::Node>::const_iterator
  VariablesDependencyGraph::nodesBegin() const
  {
    return nodes.begin();
  }

  llvm::SmallVector<VariablesDependencyGraph::Node>::const_iterator
  VariablesDependencyGraph::nodesEnd() const
  {
    return nodes.end();
  }

  bool VariablesDependencyGraph::hasCycles() const
  {
    auto beginIt = llvm::scc_begin(this);
    auto endIt =  llvm::scc_end(this);

    for (auto it = beginIt; it != endIt; ++it) {
      if (it.hasCycle()) {
        return true;
      }
    }

    return false;
  }

  llvm::SmallVector<VariableOp> VariablesDependencyGraph::postOrder() const
  {
    assert(!hasCycles());

    llvm::SmallVector<VariableOp> result;
    std::set<Node> set;

    for (const auto& node : llvm::post_order_ext(this, set)) {
      if (node != getEntryNode()) {
        result.push_back(mlir::cast<VariableOp>(node.variable));
      }
    }

    std::reverse(result.begin(), result.end());
    return result;
  }

  llvm::SmallVector<VariableOp>
  VariablesDependencyGraph::reversePostOrder() const
  {
    auto result = postOrder();
    std::reverse(result.begin(), result.end());
    return result;
  }
}
