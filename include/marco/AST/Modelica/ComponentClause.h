#ifndef MARCO_AST_MODELICA_COMPONENTCLAUSE_H
#define MARCO_AST_MODELICA_COMPONENTCLAUSE_H

#include "marco/AST/Modelica/ComponentDeclaration.h"
#include "marco/AST/Modelica/ConstrainingClause.h"
#include "marco/AST/Modelica/ElementPrefixes.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/Name.h"
#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/TypePrefix.h"

namespace marco::ast::modelica {
/// Component clause, declaring one or more components with
/// the same prefixes and type.
class ComponentClause
    : public NodeImpl<ComponentClause, Node, NodeKind::ComponentClause> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ElementPrefixes &getPrefixes();

  const ElementPrefixes &getPrefixes() const;

  TypePrefix &getTypePrefix();

  const TypePrefix &getTypePrefix() const;

  const Name &getTypeSpecifier() const;

  void setTypeSpecifier(Name name);

  /// Array subscripts of the type specifier, which apply to all the
  /// declarations (e.g., `Real[3] a, b;`).
  ChildrenRef<Expression> getSubscripts() const;

  void addSubscript(std::unique_ptr<Expression> node);

  ChildrenRef<ComponentDeclaration> getDeclarations() const;

  void addDeclaration(std::unique_ptr<ComponentDeclaration> node);

  ConstrainingClause *getConstrainingClause() const;

  void setConstrainingClause(std::unique_ptr<ConstrainingClause> node);

private:
  ElementPrefixes prefixes;
  TypePrefix typePrefix;
  Name typeSpecifier;
  Children<Expression> subscripts;
  Children<ComponentDeclaration> declarations;
  Child<ConstrainingClause> constrainingClause;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_COMPONENTCLAUSE_H
