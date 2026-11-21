#ifndef MARCO_AST_MODELICA_EXTENDSCLAUSE_H
#define MARCO_AST_MODELICA_EXTENDSCLAUSE_H

#include "marco/AST/Modelica/ClassModification.h"
#include "marco/AST/Modelica/Name.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Extends-clause. The modification may contain
/// inheritance modifications.
class ExtendsClause
    : public NodeImpl<ExtendsClause, Node, NodeKind::ExtendsClause> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  bool isProtected() const;

  void setProtected(bool value);

  const Name &getBaseClass() const;

  void setBaseClass(Name name);

  ClassModification *getClassModification() const;

  void setClassModification(std::unique_ptr<ClassModification> node);

  ClassModification *getAnnotation() const;

  void setAnnotation(std::unique_ptr<ClassModification> node);

private:
  bool protectedElement = false;
  Name baseClass;
  Child<ClassModification> classModification;
  Child<ClassModification> annotation;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_EXTENDSCLAUSE_H
