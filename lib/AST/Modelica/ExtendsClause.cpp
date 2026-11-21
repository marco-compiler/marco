#include "marco/AST/Modelica/ExtendsClause.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ExtendsClause::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(classModification, callback);
  walk(annotation, callback);
}

void ExtendsClause::addJSONProperties(llvm::json::Object &obj) const {
  if (protectedElement) {
    obj["protected"] = true;
  }

  obj["base_class"] = modelica::toJSON(baseClass);

  if (classModification) {
    obj["class_modification"] = modelica::toJSON(classModification.get());
  }

  if (annotation) {
    obj["annotation"] = modelica::toJSON(annotation.get());
  }
}

bool ExtendsClause::isProtected() const { return protectedElement; }

void ExtendsClause::setProtected(bool value) { protectedElement = value; }

const Name &ExtendsClause::getBaseClass() const { return baseClass; }

void ExtendsClause::setBaseClass(Name name) { baseClass = std::move(name); }

ClassModification *ExtendsClause::getClassModification() const {
  return classModification.get();
}

void ExtendsClause::setClassModification(
    std::unique_ptr<ClassModification> node) {
  classModification = std::move(node);
}

ClassModification *ExtendsClause::getAnnotation() const {
  return annotation.get();
}

void ExtendsClause::setAnnotation(std::unique_ptr<ClassModification> node) {
  annotation = std::move(node);
}
} // namespace marco::ast::modelica
