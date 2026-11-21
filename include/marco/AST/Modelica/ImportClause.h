#ifndef MARCO_AST_MODELICA_IMPORTCLAUSE_H
#define MARCO_AST_MODELICA_IMPORTCLAUSE_H

#include "marco/AST/Modelica/Description.h"
#include "marco/AST/Modelica/Name.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Import-clause.
class ImportClause
    : public NodeImpl<ImportClause, Node, NodeKind::ImportClause> {
public:
  enum class Form {
    /// `import A.B;`
    Qualified,
    /// `import X = A.B;`
    Renaming,
    /// `import A.B.*;`
    Unqualified,
    /// `import A.B.{C, D};`
    Multiple
  };

  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  bool isProtected() const;

  void setProtected(bool value);

  Form getForm() const;

  void setForm(Form value);

  const Name &getName() const;

  void setName(Name newName);

  llvm::StringRef getAlias() const;

  void setAlias(std::string value);

  llvm::ArrayRef<std::string> getImportedNames() const;

  void addImportedName(std::string value);

  Description &getDescription();

  const Description &getDescription() const;

private:
  bool protectedElement = false;
  Form form = Form::Qualified;
  Name name;
  std::string alias;
  std::vector<std::string> importedNames;
  Description description;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_IMPORTCLAUSE_H
