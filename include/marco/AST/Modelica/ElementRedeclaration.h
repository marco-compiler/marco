#ifndef MARCO_AST_MODELICA_ELEMENTREDECLARATION_H
#define MARCO_AST_MODELICA_ELEMENTREDECLARATION_H

#include "marco/AST/Modelica/Argument.h"
#include "marco/AST/Modelica/ConstrainingClause.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Declaration of an element within a modification, that is an
/// element-redeclaration or an element-replaceable. The
/// element is either a short class definition (ClassDefinition) or a
/// component clause with a single declaration (ComponentClause).
class ElementRedeclaration
    : public NodeImpl<ElementRedeclaration, Argument,
                      NodeKind::Argument_ElementRedeclaration> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  bool isRedeclare() const;

  void setRedeclare(bool value);

  bool isEach() const;

  void setEach(bool value);

  bool isFinal() const;

  void setFinal(bool value);

  bool isReplaceable() const;

  void setReplaceable(bool value);

  Node *getElement() const;

  void setElement(std::unique_ptr<Node> node);

  ConstrainingClause *getConstrainingClause() const;

  void setConstrainingClause(std::unique_ptr<ConstrainingClause> node);

private:
  bool redeclare = false;
  bool each = false;
  bool final = false;
  bool replaceable = false;
  Child<Node> element;
  Child<ConstrainingClause> constrainingClause;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_ELEMENTREDECLARATION_H
