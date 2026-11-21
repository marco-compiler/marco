#ifndef MARCO_AST_MODELICA_MEMBERACCESS_H
#define MARCO_AST_MODELICA_MEMBERACCESS_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Member access applied to a parenthesized expression, as in `(f(x)).a`.
class MemberAccess : public NodeImpl<MemberAccess, Expression,
                                     NodeKind::Expression_MemberAccess> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  Expression *getBase() const;

  void setBase(std::unique_ptr<Expression> node);

  llvm::StringRef getMember() const;

  void setMember(std::string name);

private:
  Child<Expression> base;
  std::string member;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_MEMBERACCESS_H
