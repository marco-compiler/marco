#include "marco/AST/BaseModelica/Node/Class.h"
#include "marco/AST/BaseModelica/Node/Algorithm.h"
#include "marco/AST/BaseModelica/Node/Annotation.h"
#include "marco/AST/BaseModelica/Node/EquationSection.h"
#include "marco/AST/BaseModelica/Node/ExternalFunctionCall.h"
#include "marco/AST/BaseModelica/Node/Member.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast {
Class::Class(const Class &other)
    : ASTNode(other), name(other.name), external(other.external) {
  setVariables(other.variables);
  setEquationSections(other.equationSections);
  setAlgorithms(other.algorithms);
  setInnerClasses(other.innerClasses);
  setExternalLanguage(other.externalLanguage);

  if (other.hasExternalFunctionCall()) {
    setExternalFunctionCall(other.externalFunctionCall->clone());
  }

  if (other.hasExternalAnnotation()) {
    setExternalAnnotation(other.externalAnnotation->clone());
  }

  if (other.hasAnnotation()) {
    setAnnotation(other.annotation->clone());
  }
}

Class::~Class() = default;

void Class::addJSONProperties(llvm::json::Object &obj) const {
  obj["name"] = getName();

  llvm::SmallVector<llvm::json::Value> variablesJson;

  for (const auto &variable : variables) {
    variablesJson.push_back(variable->toJSON());
  }

  obj["variables"] = llvm::json::Array(variablesJson);

  llvm::SmallVector<llvm::json::Value> equationSectionsJson;

  for (const auto &section : equationSections) {
    equationSectionsJson.push_back(section->toJSON());
  }

  obj["equation_sections"] = llvm::json::Array(equationSectionsJson);

  llvm::SmallVector<llvm::json::Value> algorithmsJson;

  for (const auto &algorithm : algorithms) {
    algorithmsJson.push_back(algorithm->toJSON());
  }

  obj["algorithms"] = llvm::json::Array(algorithmsJson);

  llvm::SmallVector<llvm::json::Value> innerClassesJson;

  for (const auto &innerClass : innerClasses) {
    innerClassesJson.push_back(innerClass->toJSON());
  }

  obj["inner_classes"] = llvm::json::Array(innerClassesJson);
  obj["external"] = isExternal();

  if (isExternal()) {
    obj["external_language"] = getExternalLanguage();

    if (hasExternalFunctionCall()) {
      obj["external_function_call"] = getExternalFunctionCall()->toJSON();
    }

    if (hasExternalAnnotation()) {
      obj["external_annotation"] = getExternalAnnotation()->toJSON();
    }
  }

  ASTNode::addJSONProperties(obj);
}

llvm::StringRef Class::getName() const { return name; }

void Class::setName(llvm::StringRef newName) { name = newName.str(); }

llvm::ArrayRef<std::unique_ptr<ASTNode>> Class::getVariables() const {
  return variables;
}

void Class::setVariables(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> newVariables) {
  variables.clear();

  for (const auto &variable : newVariables) {
    assert(variable->isa<Member>());
    auto &clone = variables.emplace_back(variable->clone());
    clone->setParent(this);
  }
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> Class::getEquationSections() const {
  return equationSections;
}

void Class::setEquationSections(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> newEquationSections) {
  equationSections.clear();

  for (const auto &equationSection : newEquationSections) {
    assert(equationSection->isa<EquationSection>());
    auto &clone = equationSections.emplace_back(equationSection->clone());
    clone->setParent(this);
  }
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> Class::getAlgorithms() const {
  return algorithms;
}

void Class::setAlgorithms(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> newAlgorithms) {
  algorithms.clear();

  for (const auto &algorithm : newAlgorithms) {
    assert(algorithm->isa<Algorithm>());
    auto &clone = algorithms.emplace_back(algorithm->clone());
    clone->setParent(this);
  }
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> Class::getInnerClasses() const {
  return innerClasses;
}

void Class::setInnerClasses(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> newInnerClasses) {
  innerClasses.clear();

  for (const auto &cls : newInnerClasses) {
    assert(cls->isa<Class>());
    auto &clone = innerClasses.emplace_back(cls->clone());
    clone->setParent(this);
  }
}

bool Class::isExternal() const { return external; }

void Class::setExternal(bool isExternal) { external = isExternal; }

llvm::StringRef Class::getExternalLanguage() const { return externalLanguage; }

void Class::setExternalLanguage(llvm::StringRef language) {
  externalLanguage = language.str();
}

bool Class::hasExternalFunctionCall() const {
  return externalFunctionCall != nullptr;
}

ExternalFunctionCall *Class::getExternalFunctionCall() {
  assert(externalFunctionCall != nullptr && "External function call not set");
  return externalFunctionCall->cast<ExternalFunctionCall>();
}

const ExternalFunctionCall *Class::getExternalFunctionCall() const {
  assert(externalFunctionCall != nullptr && "External function call not set");
  return externalFunctionCall->cast<ExternalFunctionCall>();
}

void Class::setExternalFunctionCall(std::unique_ptr<ASTNode> node) {
  assert(node->isa<ExternalFunctionCall>());
  externalFunctionCall = std::move(node);
  externalFunctionCall->setParent(this);
}

bool Class::hasExternalAnnotation() const {
  return externalAnnotation != nullptr;
}

Annotation *Class::getExternalAnnotation() {
  assert(externalAnnotation != nullptr && "External annotation not set");
  return externalAnnotation->cast<Annotation>();
}

const Annotation *Class::getExternalAnnotation() const {
  assert(externalAnnotation != nullptr && "External annotation not set");
  return externalAnnotation->cast<Annotation>();
}

void Class::setExternalAnnotation(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Annotation>());
  externalAnnotation = std::move(node);
  externalAnnotation->setParent(this);
}

bool Class::hasAnnotation() const { return annotation != nullptr; }

Annotation *Class::getAnnotation() {
  assert(annotation != nullptr && "Annotation not set");
  return annotation->cast<Annotation>();
}

const Annotation *Class::getAnnotation() const {
  assert(annotation != nullptr && "Annotation not set");
  return annotation->cast<Annotation>();
}

void Class::setAnnotation(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Annotation>());
  annotation = std::move(node);
  annotation->setParent(this);
}
} // namespace marco::ast
