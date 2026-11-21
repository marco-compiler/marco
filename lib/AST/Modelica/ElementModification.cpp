#include "marco/AST/Modelica/ElementModification.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ElementModification::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(modification, callback);
}

void ElementModification::addJSONProperties(llvm::json::Object &obj) const {
  if (each) {
    obj["each"] = true;
  }

  if (final) {
    obj["final"] = true;
  }

  obj["name"] = modelica::toJSON(name);

  if (modification) {
    obj["modification"] = modelica::toJSON(modification.get());
  }

  if (!description.empty()) {
    obj["description"] = description;
  }
}

bool ElementModification::isEach() const { return each; }

void ElementModification::setEach(bool value) { each = value; }

bool ElementModification::isFinal() const { return final; }

void ElementModification::setFinal(bool value) { final = value; }

const Name &ElementModification::getName() const { return name; }

void ElementModification::setName(Name newName) { name = std::move(newName); }

Modification *ElementModification::getModification() const {
  return modification.get();
}

void ElementModification::setModification(std::unique_ptr<Modification> node) {
  modification = std::move(node);
}

llvm::StringRef ElementModification::getDescription() const {
  return description;
}

void ElementModification::setDescription(std::string text) {
  description = std::move(text);
}
} // namespace marco::ast::modelica
