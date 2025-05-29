#ifndef PUBLIC_MARCO_AST_NODE_RETURNSTATEMENT_H
#define PUBLIC_MARCO_AST_NODE_RETURNSTATEMENT_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Statement.h"
#include "marco/Parser/Location.h"
#include <llvm/Support/JSON.h>
#include <memory>

namespace marco::ast {
class ReturnStatement : public Statement {
public:
  explicit ReturnStatement(SourceRange location);

  ReturnStatement(const ReturnStatement &other);

  ~ReturnStatement() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Statement_Return;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;
};
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_RETURNSTATEMENT_H
