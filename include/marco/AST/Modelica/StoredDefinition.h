#ifndef MARCO_AST_MODELICA_STOREDDEFINITION_H
#define MARCO_AST_MODELICA_STOREDDEFINITION_H

#include "marco/AST/Modelica/ClassDefinition.h"
#include "marco/AST/Modelica/Name.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Content of a file: an optional within-clause and the
/// class definitions.
class StoredDefinition
    : public NodeImpl<StoredDefinition, Node, NodeKind::StoredDefinition> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  /// Get the package specified by the within-clause, if any. An empty name
  /// means the top-level (`within;`).
  const std::optional<Name> &getWithin() const;

  void setWithin(std::optional<Name> name);

  ChildrenRef<ClassDefinition> getClasses() const;

  void addClass(std::unique_ptr<ClassDefinition> node);

private:
  std::optional<Name> within;
  Children<ClassDefinition> classes;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_STOREDDEFINITION_H
