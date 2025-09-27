#ifndef MARCO_AST_BASEMODELICA_NODE_BREAKSTATEMENT_H
#define MARCO_AST_BASEMODELICA_NODE_BREAKSTATEMENT_H

#include "marco/AST/BaseModelica/Node/Statement.h"

namespace marco::ast::bmodelica {
class BreakStatement : public Statement {
public:
  explicit BreakStatement(SourceRange location);

  BreakStatement(const BreakStatement &other);

  ~BreakStatement() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Statement_Break;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_BREAKSTATEMENT_H
