#ifndef MARCO_AST_BASEMODELICA_CALLEQUATION_H
#define MARCO_AST_BASEMODELICA_CALLEQUATION_H

#include "marco/AST/BaseModelica/Equation.h"

namespace marco::ast::bmodelica {
class Expression;

class CallEquation : public Equation {
public:
  explicit CallEquation(SourceRange location);

  CallEquation(const CallEquation &other);

  ~CallEquation() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() == ASTNodeKind::Equation_Call;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getCall();

  const Expression *getCall() const;

  void setCall(std::unique_ptr<ASTNode> node);

private:
  std::unique_ptr<ASTNode> call;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_CALLEQUATION_H
