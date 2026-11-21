#ifndef MARCO_AST_MODELICA_COMPONENTDECLARATION_H
#define MARCO_AST_MODELICA_COMPONENTDECLARATION_H

#include "marco/AST/Modelica/Description.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/Modification.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Single declaration of a component clause (`x[2](start = 1) = 3 if c`).
class ComponentDeclaration
    : public NodeImpl<ComponentDeclaration, Node,
                      NodeKind::ComponentDeclaration> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  llvm::StringRef getName() const;

  void setName(std::string newName);

  ChildrenRef<Expression> getSubscripts() const;

  void addSubscript(std::unique_ptr<Expression> node);

  Modification *getModification() const;

  void setModification(std::unique_ptr<Modification> node);

  /// Get the condition of a conditional component.
  Expression *getCondition() const;

  void setCondition(std::unique_ptr<Expression> node);

  Description &getDescription();

  const Description &getDescription() const;

private:
  std::string name;
  Children<Expression> subscripts;
  Child<Modification> modification;
  Child<Expression> condition;
  Description description;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_COMPONENTDECLARATION_H
