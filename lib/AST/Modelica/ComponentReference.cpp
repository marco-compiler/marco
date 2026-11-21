#include "marco/AST/Modelica/ComponentReference.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ComponentReference::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  for (const Part &part : parts) {
    walk(part.subscripts, callback);
  }
}

void ComponentReference::addJSONProperties(llvm::json::Object &obj) const {
  obj["global"] = global;
  llvm::json::Array array;

  for (const Part &part : parts) {
    llvm::json::Object partObj;
    partObj["name"] = part.name;

    if (!part.subscripts.empty()) {
      partObj["subscripts"] = modelica::toJSON(part.subscripts);
    }

    array.push_back(std::move(partObj));
  }

  obj["parts"] = std::move(array);
}

bool ComponentReference::isGlobal() const { return global; }

void ComponentReference::setGlobal(bool value) { global = value; }

llvm::ArrayRef<ComponentReference::Part> ComponentReference::getParts() const {
  return parts;
}

llvm::MutableArrayRef<ComponentReference::Part> ComponentReference::getParts() {
  return parts;
}

void ComponentReference::addPart(Part part) {
  parts.push_back(std::move(part));
}

Name ComponentReference::getName() const {
  Name result;
  result.global = global;
  result.location = getLocation();

  for (const Part &part : parts) {
    result.parts.push_back(part.name);
  }

  return result;
}
} // namespace marco::ast::modelica
