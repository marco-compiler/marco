#include "marco/AST/Modelica/ConstrainingClause.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ConstrainingClause::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(classModification, callback);
  walk(annotation, callback);
}

void ConstrainingClause::addJSONProperties(llvm::json::Object &obj) const {
  obj["type"] = modelica::toJSON(typeSpecifier);

  if (classModification) {
    obj["class_modification"] = modelica::toJSON(classModification.get());
  }

  if (!description.empty()) {
    obj["description"] = description;
  }

  if (annotation) {
    obj["annotation"] = modelica::toJSON(annotation.get());
  }
}

const Name &ConstrainingClause::getTypeSpecifier() const {
  return typeSpecifier;
}

void ConstrainingClause::setTypeSpecifier(Name name) {
  typeSpecifier = std::move(name);
}

ClassModification *ConstrainingClause::getClassModification() const {
  return classModification.get();
}

void ConstrainingClause::setClassModification(
    std::unique_ptr<ClassModification> node) {
  classModification = std::move(node);
}

llvm::StringRef ConstrainingClause::getDescription() const {
  return description;
}

void ConstrainingClause::setDescription(std::string text) {
  description = std::move(text);
}

ClassModification *ConstrainingClause::getAnnotation() const {
  return annotation.get();
}

void ConstrainingClause::setAnnotation(
    std::unique_ptr<ClassModification> node) {
  annotation = std::move(node);
}
} // namespace marco::ast::modelica
