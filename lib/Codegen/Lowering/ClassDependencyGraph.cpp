#include "marco/Codegen/Lowering/ClassDependencyGraph.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"

namespace marco::codegen::lowering {
ClassDependencyGraph::Node::Node() : graph(nullptr) {}

ClassDependencyGraph::Node::Node(const ClassDependencyGraph *graph,
                                 ClassPath path)
    : graph(graph), path(std::move(path)) {}

bool ClassDependencyGraph::Node::operator==(const Node &other) const {
  return graph == other.graph && path == other.path;
}

bool ClassDependencyGraph::Node::operator!=(const Node &other) const {
  return graph != other.graph || path != other.path;
}

bool ClassDependencyGraph::Node::operator<(const Node &other) const {
  if (path.size() == 0) {
    return true;
  }

  if (other.path.size() == 0) {
    return false;
  }

  return path < other.path;
}

ClassDependencyGraph::ClassDependencyGraph() {
  // Entry node, which is connected to every other node.
  nodes.emplace_back(this, ClassPath());

  // Ensure that the set of children for the entry node exists, even in
  // case of no other nodes.
  arcs[getEntryNode().path] = {};
}

ClassDependencyGraph::~ClassDependencyGraph() = default;

size_t ClassDependencyGraph::getNumOfNodes() const { return nodes.size(); }

const ClassDependencyGraph::Node &ClassDependencyGraph::getEntryNode() const {
  return nodes[0];
}

std::set<ClassDependencyGraph::Node>::const_iterator
ClassDependencyGraph::childrenBegin(ClassDependencyGraph::Node node) const {
  auto it = arcs.find(node.path);
  assert(it != arcs.end());
  const auto &children = it->second;
  return children.begin();
}

std::set<ClassDependencyGraph::Node>::const_iterator
ClassDependencyGraph::childrenEnd(ClassDependencyGraph::Node node) const {
  auto it = arcs.find(node.path);
  assert(it != arcs.end());
  const auto &children = it->second;
  return children.end();
}

llvm::SmallVector<ClassDependencyGraph::Node>::const_iterator
ClassDependencyGraph::nodesBegin() const {
  return nodes.begin();
}

llvm::SmallVector<ClassDependencyGraph::Node>::const_iterator
ClassDependencyGraph::nodesEnd() const {
  return nodes.end();
}

llvm::SmallVector<llvm::DenseSet<ClassPath>>
ClassDependencyGraph::getCycles() const {
  llvm::SmallVector<llvm::DenseSet<ClassPath>> result;

  auto beginIt = llvm::scc_begin(this);
  auto endIt = llvm::scc_end(this);

  for (auto it = beginIt; it != endIt; ++it) {
    if (it.hasCycle()) {
      llvm::DenseSet<ClassPath> scc;

      for (const Node &node : *it) {
        scc.insert(node.path);
      }

      result.push_back(std::move(scc));
    }
  }

  return result;
}

llvm::SmallVector<ClassPath> ClassDependencyGraph::postOrder() const {
  // The graph must have no cycles.
  assert(getCycles().empty());

  llvm::SmallVector<ClassPath> result;
  std::set<Node> set;

  for (const Node &node : llvm::post_order_ext(this, set)) {
    if (node != getEntryNode()) {
      result.push_back(node.path);
    }
  }

  return result;
}

llvm::SmallVector<ClassPath> ClassDependencyGraph::reversePostOrder() const {
  auto result = postOrder();
  std::reverse(result.begin(), result.end());
  return result;
}
} // namespace marco::codegen::lowering
