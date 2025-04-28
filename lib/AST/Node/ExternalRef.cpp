#include "marco/AST/Node/ExternalRef.h"
#include "marco/AST/Node/ExternalFunctionCall.h"
#include "marco/AST/Node/Annotation.h"


using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
ExtenralRef::ExternalRef(SourceRange location) : ASTNode(ASTNode::Kind::External_Ref, std::move(location)) {}

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

void ExternalRef::setLanguageSpecification(llvm::StringRef newLanguageSpecification) {
  languageSpecification = newLanguageSpecification.str(); 
}

llvm::StringRef ExternalRef::getLanguageSpecification() {
  return languageSpecification; 
}

void ExternalRef::setExternalFunctionCall(std::unique_ptr<ASTNode> externalFunctionCall) {
  assert(node->isa<ExternalFunctionCall>()); 
  externalFunctionCall = std::move(externalFunctionCall); 
  externalFunctionCall -> setParent(this); 
} 

const ExternalFunctionCall *ExternalRef::getExternalFunctionCall() const{
  return externalFunctionCall->cast<ExternalFunctionCall>();
} 

ExternalFunctionCall *ExternalRef::getExternalFunctionCall() {
  return externalFunctionCall->cast<ExternalFunctionCall>();
}

void ExternalRef::setAnnotationClause(std::unique_ptr<ASTNode> annotationClause){
  assert(node->isa<Annotation>()); 
  annotationClause = std::move(annotationClause); 
  annotationClause -> setParent(this); 
}

const Annotation *ExternalRef::getAnnotationClause() const {
  return annotationClause->cast<Annotation>();
}

Annotation *ExternalRef::getAnnotationClause(){
  return annotationClause->cast<Annotation>();
} 
} // namespace marco::ast
