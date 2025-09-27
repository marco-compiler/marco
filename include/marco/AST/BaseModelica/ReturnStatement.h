#ifndef MARCO_AST_BASEMODELICA_RETURNSTATEMENT_H
#define MARCO_AST_BASEMODELICA_RETURNSTATEMENT_H

#include "marco/AST/BaseModelica/Statement.h"

namespace marco::ast::bmodelica {
class ReturnStatement : public Statement {
public:
  explicit ReturnStatement(SourceRange location);

  ReturnStatement(const ReturnStatement &other);

  ~ReturnStatement() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() == ASTNodeKind::Statement_Return;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_RETURNSTATEMENT_H
