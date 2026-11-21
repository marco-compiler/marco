#ifndef MARCO_AST_MODELICA_ENUMERATIONLITERAL_H
#define MARCO_AST_MODELICA_ENUMERATIONLITERAL_H

#include "marco/AST/Modelica/Description.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Literal of an enumeration type.
class EnumerationLiteral
    : public NodeImpl<EnumerationLiteral, Node, NodeKind::EnumerationLiteral> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  llvm::StringRef getName() const;

  void setName(std::string newName);

  Description &getDescription();

  const Description &getDescription() const;

private:
  std::string name;
  Description description;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_ENUMERATIONLITERAL_H
