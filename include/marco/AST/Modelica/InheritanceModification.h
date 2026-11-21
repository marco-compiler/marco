#ifndef MARCO_AST_MODELICA_INHERITANCEMODIFICATION_H
#define MARCO_AST_MODELICA_INHERITANCEMODIFICATION_H

#include "marco/AST/Modelica/Argument.h"
#include "marco/AST/Modelica/ComponentReference.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Deselection of selective model extension (`break name` or
/// `break connect(a, b)`).
class InheritanceModification
    : public NodeImpl<InheritanceModification, Argument,
                      NodeKind::Argument_InheritanceModification> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  /// Name of the deselected component, empty for connection deselections.
  llvm::StringRef getComponent() const;

  void setComponent(std::string name);

  /// Connected elements of a connection deselection.
  ComponentReference *getConnectLhs() const;

  ComponentReference *getConnectRhs() const;

  void setConnect(std::unique_ptr<ComponentReference> lhs,
                  std::unique_ptr<ComponentReference> rhs);

  bool isConnect() const;

private:
  std::string component;
  Child<ComponentReference> connectLhs;
  Child<ComponentReference> connectRhs;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_INHERITANCEMODIFICATION_H
