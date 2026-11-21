#include "marco/AST/Modelica/ImportClause.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ImportClause::walkChildren(llvm::function_ref<void(Node *)> callback) {
  description.walkChildren(callback);
}

void ImportClause::addJSONProperties(llvm::json::Object &obj) const {
  if (protectedElement) {
    obj["protected"] = true;
  }

  obj["name"] = modelica::toJSON(name);

  switch (form) {
  case Form::Qualified:
    obj["form"] = "qualified";
    break;

  case Form::Renaming:
    obj["form"] = "renaming";
    obj["alias"] = alias;
    break;

  case Form::Unqualified:
    obj["form"] = "unqualified";
    break;

  case Form::Multiple: {
    obj["form"] = "multiple";
    llvm::json::Array names;

    for (const std::string &importedName : importedNames) {
      names.push_back(importedName);
    }

    obj["names"] = std::move(names);
    break;
  }
  }

  description.addToJSON(obj);
}

bool ImportClause::isProtected() const { return protectedElement; }

void ImportClause::setProtected(bool value) { protectedElement = value; }

ImportClause::Form ImportClause::getForm() const { return form; }

void ImportClause::setForm(Form value) { form = value; }

const Name &ImportClause::getName() const { return name; }

void ImportClause::setName(Name newName) { name = std::move(newName); }

llvm::StringRef ImportClause::getAlias() const { return alias; }

void ImportClause::setAlias(std::string value) { alias = std::move(value); }

llvm::ArrayRef<std::string> ImportClause::getImportedNames() const {
  return importedNames;
}

void ImportClause::addImportedName(std::string value) {
  importedNames.push_back(std::move(value));
}

Description &ImportClause::getDescription() { return description; }

const Description &ImportClause::getDescription() const { return description; }
} // namespace marco::ast::modelica
