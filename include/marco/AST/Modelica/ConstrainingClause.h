#ifndef MARCO_AST_MODELICA_CONSTRAININGCLAUSE_H
#define MARCO_AST_MODELICA_CONSTRAININGCLAUSE_H

#include "marco/AST/Modelica/ClassModification.h"
#include "marco/AST/Modelica/Name.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Constraining clause (`constrainedby Type(modification) "description"`).
class ConstrainingClause
    : public NodeImpl<ConstrainingClause, Node, NodeKind::ConstrainingClause> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  const Name &getTypeSpecifier() const;

  void setTypeSpecifier(Name name);

  ClassModification *getClassModification() const;

  void setClassModification(std::unique_ptr<ClassModification> node);

  llvm::StringRef getDescription() const;

  void setDescription(std::string text);

  ClassModification *getAnnotation() const;

  void setAnnotation(std::unique_ptr<ClassModification> node);

private:
  Name typeSpecifier;
  Child<ClassModification> classModification;
  std::string description;
  Child<ClassModification> annotation;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_CONSTRAININGCLAUSE_H
