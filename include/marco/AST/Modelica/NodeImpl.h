#ifndef MARCO_AST_MODELICA_NODEIMPL_H
#define MARCO_AST_MODELICA_NODEIMPL_H

#include "marco/AST/Modelica/Node.h"
#include <memory>

namespace marco::ast::modelica {
/// Helper class implementing the methods common to all the concrete nodes.
template <typename Derived, typename Base, NodeKind Kind>
class NodeImpl : public Base {
public:
  explicit NodeImpl(SourceRange location) : Base(Kind, location) {}

  static bool classof(const ASTNode *node) {
    return node->getKind<NodeKind>() == Kind;
  }

  std::unique_ptr<ASTNode> clone() const override {
    auto result =
        std::make_unique<Derived>(static_cast<const Derived &>(*this));

    result->setParent(nullptr);
    result->setParentOfChildren();
    return result;
  }
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_NODEIMPL_H
