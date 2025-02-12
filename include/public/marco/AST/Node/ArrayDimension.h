#ifndef MARCO_AST_NODE_ARRAYDIMENSION_H
#define MARCO_AST_NODE_ARRAYDIMENSION_H

#include "marco/AST/Node/ASTNode.h"
#include <variant>

namespace marco::ast {
class Expression;

/// Represent the size of an array dimension.
/// Can be either static or determined by an expression. Note that
/// a dynamic size (":", in Modelica) is considered static and is
/// represented by value "-1".
class ArrayDimension : public ASTNode {
public:
  static constexpr long kDynamicSize = -1;

  ArrayDimension(SourceRange location);

  ArrayDimension(const ArrayDimension &other);

  ~ArrayDimension();

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::ArrayDimension;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  template <class Visitor>
  auto visit(Visitor &&visitor) {
    return std::visit(std::forward<Visitor>(visitor), size);
  }

  template <class Visitor>
  auto visit(Visitor &&visitor) const {
    return std::visit(std::forward<Visitor>(visitor), size);
  }

  [[nodiscard]] bool hasExpression() const;

  [[nodiscard]] bool isDynamic() const;

  [[nodiscard]] long getNumericSize() const;

  Expression *getExpression();

  const Expression *getExpression() const;

  void setSize(int64_t value);

  void setSize(std::unique_ptr<ASTNode> node);

private:
  std::variant<int64_t, std::unique_ptr<ASTNode>> size;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_ARRAYDIMENSION_H
