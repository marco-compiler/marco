#include "marco/AST/Analysis/DynamicDimensionsGraph.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include <stack>

using namespace ::marco::ast;

namespace
{
  struct ExpressionVisitor
  {
    std::set<const Expression*> operator()(const Array& array)
    {
      std::set<const Expression*> result;

      for (const auto& value : array) {
        result.insert(value.get());
      }

      return result;
    }

    std::set<const Expression*> operator()(const Call& call)
    {
      std::set<const Expression*> result;

      for (const auto& expression : call.getArgs()) {
        result.insert(expression.get());
      }

      return result;
    }

    std::set<const Expression*> operator()(const Constant& constant)
    {
      return {};
    }

    std::set<const Expression*> operator()(const Operation& operation)
    {
      std::set<const Expression*> result;

      for (const auto& expression : operation.getArguments()) {
        result.insert(expression.get());
      }

      return result;
    }

    std::set<const Expression*> operator()(
        const ReferenceAccess& referenceAccess)
    {
      return {};
    }

    std::set<const Expression*> operator()(const Tuple& tuple)
    {
      std::set<const Expression*> result;

      for (const auto& expression : tuple) {
        result.insert(expression.get());
      }

      return result;
    }

    std::set<const Expression*> operator()(
        const RecordInstance& recordInstance)
    {
      std::set<const Expression*> result;

      for (const auto& expression : recordInstance) {
        result.insert(expression.get());
      }

      return result;
    }
  };
}

namespace marco::ast
{
  DynamicDimensionsGraph::Node::Node()
      : graph(nullptr), member(nullptr)
  {
  }

  DynamicDimensionsGraph::Node::Node(
      const DynamicDimensionsGraph* graph, const Member* member)
      : graph(graph), member(member)
  {
  }

  bool DynamicDimensionsGraph::Node::operator==(
      const DynamicDimensionsGraph::Node& other) const
  {
    return graph == other.graph && member == other.member;
  }

  bool DynamicDimensionsGraph::Node::operator!=(
      const DynamicDimensionsGraph::Node& other) const
  {
    return graph != other.graph || member != other.member;
  }

  bool DynamicDimensionsGraph::Node::operator<(
      const DynamicDimensionsGraph::Node& other) const
  {
    if (member == nullptr) {
      return true;
    }

    if (other.member == nullptr) {
      return false;
    }

    return member < other.member;
  }

  DynamicDimensionsGraph::DynamicDimensionsGraph()
  {
    // Entry node, which is connected to every other node.
    nodes.emplace_back(this, nullptr);
  }

  void DynamicDimensionsGraph::addMembersGroup(
      llvm::ArrayRef<const Member*> group)
  {
    for (const Member* member : group) {
      assert(member != nullptr);

      Node node(this, member);
      nodes.push_back(node);
      symbolTable[member->getName()] = node;

      // Connect the entry node.
      arcs[getEntryNode().member].insert(node);
    }

    for (size_t i = 1, e = group.size(); i < e; ++i) {
      arcs[group[i]].insert(Node(this, group[i - 1]));
    }
  }

  void DynamicDimensionsGraph::discoverDependencies()
  {
    for (const Node& node : nodes) {
      if (node != getEntryNode()) {
        const Member* member = node.member;

        auto& children = arcs[member];

        for (const auto& dimension : member->getType().getDimensions()) {
          if (dimension.hasExpression()) {
            const Expression* expression = dimension.getExpression();
            std::set<const Member*> usedMembers = getUsedMembers(expression);

            for (const Member* dependency : usedMembers) {
              children.insert(Node(this, dependency));
            }
          }
        }
      }
    }
  }

  size_t DynamicDimensionsGraph::getNumOfNodes() const
  {
    return nodes.size();
  }

  DynamicDimensionsGraph::Node DynamicDimensionsGraph::getEntryNode() const
  {
    return nodes[0];
  }

  std::set<DynamicDimensionsGraph::Node>::const_iterator
  DynamicDimensionsGraph::childrenBegin(Node node) const
  {
    auto it = arcs.find(node.member);
    assert(it != arcs.end());
    const auto& children = it->second;
    return children.begin();
  }

  std::set<DynamicDimensionsGraph::Node>::const_iterator
  DynamicDimensionsGraph::childrenEnd(Node node) const
  {
    auto it = arcs.find(node.member);
    assert(it != arcs.end());
    const auto& children = it->second;
    return children.end();
  }

  llvm::SmallVector<DynamicDimensionsGraph::Node>::const_iterator
  DynamicDimensionsGraph::nodesBegin() const
  {
    return nodes.begin();
  }

  llvm::SmallVector<DynamicDimensionsGraph::Node>::const_iterator
  DynamicDimensionsGraph::nodesEnd() const
  {
    return nodes.end();
  }

  bool DynamicDimensionsGraph::hasCycles() const
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

  llvm::SmallVector<const Member*> DynamicDimensionsGraph::postOrder() const
  {
    llvm::SmallVector<const Member*> result;
    std::set<Node> set;

    for (const auto& node : llvm::post_order_ext(this, set)) {
      if (node != getEntryNode()) {
        result.push_back(node.member);
      }
    }

    return result;
  }

  std::set<const Member*> DynamicDimensionsGraph::getUsedMembers(
      const Expression* expression)
  {
    std::set<const Member*> result;

    std::stack<const Expression*> expressions;
    expressions.push(expression);

    while (!expressions.empty()) {
      const Expression* current = expressions.top();
      expressions.pop();

      if (auto* reference = current->dyn_get<ReferenceAccess>()) {
        const Member* member =
            symbolTable.find(reference->getName())->second.member;

        assert(member != nullptr);
        result.insert(member);
      } else {
        ExpressionVisitor visitor;

        for (const Expression* child : current->visit(visitor)) {
          expressions.push(child);
        }
      }
    }

    return result;
  }
}
