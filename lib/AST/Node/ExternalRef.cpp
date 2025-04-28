#include "marco/AST/Node/ExternalRef.h"
#include "marco/AST/Node/ExternalFunctionCall.h"
#include "marco/AST/Node/Annotation.h"


using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
ExternalRef::ExternalRef(SourceRange location) : ASTNode(ASTNode::Kind::External_Ref, std::move(location)) {}

ExternalRef::ExternalRef(const ExternalRef &other) : ASTNode(other) {
  if (other.hasLanguageSpecification()) {
    setLanguageSpecification(other.languageSpecification); 
  }
  if (other.hasExternalFunctionCall()) {
    setExternalFunctionCall(other.externalFunctionCall->clone()); 
  }
  if (other.hasAnnotationClause()) {
    setAnnotationClause(other.annotationClause->clone());
  }
}

ExternalRef::~ExternalRef() = default;

std::unique_ptr<ASTNode> ExternalRef::clone() const {
  return std::make_unique<ExternalRef>(*this);
}

llvm::json::Value ExternalRef::toJSON() const {
  llvm::json::Object result;
  result["language_specification"] = getLanguageSpecification();
  result["external_function_call"] = getExternalFunctionCall() -> toJSON();
  result["annotation_clause"] = getAnnotationClause() -> toJSON();
  addJSONProperties(result);
  return result;
}

void ExternalRef::setLanguageSpecification(std::string newLanguageSpecification) {
  languageSpecification = newLanguageSpecification; 
}

std::string ExternalRef::getLanguageSpecification() const {
  return languageSpecification; 
}

void ExternalRef::setExternalFunctionCall(std::unique_ptr<ASTNode> node) {
  assert(node->isa<ExternalFunctionCall>()); 
  externalFunctionCall = std::move(node); 
  externalFunctionCall -> setParent(this); 
} 

const ExternalFunctionCall *ExternalRef::getExternalFunctionCall() const{
  return externalFunctionCall->cast<ExternalFunctionCall>();
} 

ExternalFunctionCall *ExternalRef::getExternalFunctionCall() {
  return externalFunctionCall->cast<ExternalFunctionCall>();
}

void ExternalRef::setAnnotationClause(std::unique_ptr<ASTNode> node){
  assert(node->isa<Annotation>()); 
  annotationClause = std::move(node); 
  annotationClause -> setParent(this); 
}

const Annotation *ExternalRef::getAnnotationClause() const {
  return annotationClause->cast<Annotation>();
}

Annotation *ExternalRef::getAnnotationClause(){
  return annotationClause->cast<Annotation>();
} 

bool ExternalRef::hasLanguageSpecification() const {return languageSpecification != nullptr;}
bool ExternalRef::hasExternalFunctionCall() const {return externalFunctionCall != nullptr;}
bool ExternalRef::hasAnotationClause() const {return annotationClause != nullptr;}

} // namespace marco::ast
