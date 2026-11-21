#ifndef MARCO_AST_MODELICA_ELEMENTMODIFICATION_H
#define MARCO_AST_MODELICA_ELEMENTMODIFICATION_H

#include "marco/AST/Modelica/Argument.h"
#include "marco/AST/Modelica/Modification.h"
#include "marco/AST/Modelica/Name.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Modification of an element (`each final a.b(c = 1) = 2 "description"`).
class ElementModification
    : public NodeImpl<ElementModification, Argument,
                      NodeKind::Argument_ElementModification> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  bool isEach() const;

  void setEach(bool value);

  bool isFinal() const;

  void setFinal(bool value);

  const Name &getName() const;

  void setName(Name newName);

  Modification *getModification() const;

  void setModification(std::unique_ptr<Modification> node);

  llvm::StringRef getDescription() const;

  void setDescription(std::string text);

private:
  bool each = false;
  bool final = false;
  Name name;
  Child<Modification> modification;
  std::string description;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_ELEMENTMODIFICATION_H
