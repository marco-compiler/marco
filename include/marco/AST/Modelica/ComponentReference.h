#ifndef MARCO_AST_MODELICA_COMPONENTREFERENCE_H
#define MARCO_AST_MODELICA_COMPONENTREFERENCE_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/Name.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Component reference (e.g., `a.b[1, :].c`). Each part has a name and
/// optional subscripts, which are expressions possibly including `Colon` and
/// `End` nodes.
class ComponentReference
    : public NodeImpl<ComponentReference, Expression,
                      NodeKind::Expression_ComponentReference> {
public:
  struct Part {
    std::string name;
    Children<Expression> subscripts;
    SourceRange location = SourceRange::unknown();
  };

  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  bool isGlobal() const;

  void setGlobal(bool value);

  llvm::ArrayRef<Part> getParts() const;

  llvm::MutableArrayRef<Part> getParts();

  void addPart(Part part);

  /// Get the name made of the parts, ignoring the subscripts.
  Name getName() const;

private:
  bool global = false;
  std::vector<Part> parts;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_COMPONENTREFERENCE_H
