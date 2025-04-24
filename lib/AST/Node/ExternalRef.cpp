#include "marco/AST/Node/ExternalRef.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
ExternalRef(SourceRange location) : ASTNode(ASTNode::Kind::External_Ref, std::move(location)) {}

~ExternalRef() = default;

std::unique_ptr<ASTNode> clone() const {
  return std::make_unique<ExternalRef>(*this);
}

llvm::json::Value toJSON() const {
  llvm::json::Object result;
  result["language_specification"] = languageSpecification;
  result["external_function_call"] = getExternalFunctionCallPtr() -> toJSON();
  result["annotation_clause"] = getAnnotationPtr() -> toJSON();
  addJSONProperties(result);
  return result;
}

void setLanguageSpecification(llvm::StringRef languageSpecification) {
  languageSpecification = languageSpecification.str(); 
}

llvm::StringRef getLanguageSpecification() {
  return languageSpecification; 
}

void setExternalFunctionCall(std::unique_ptr<ASTNode> externalFunctionCall) {
  externalFunctionCall = std::move(externalFunctionCall); 
  externalFunctionCall -> setParent(this); 
} 

std::unique_ptr<ASTNode> getExternalFunctionCall(){
  return externalFunctionCall; 
}

ExternalFunctionCall *getExternalFunctionCallPtr() {
  return externalFunctionCall->cast<ExternalFunctionCall>();
}

void setAnnotationClause(std::unique_ptr<ASTNode> annotationClause){
  annotationClause = std::move(annotationClause); 
  annotationClause -> setParent(this); 
}

std::unique_ptr<ASTNode> getAnnotationClause() {
  return annotationClause; 
}

Annotation *getAnnotationPtr() {
  return annotationClause->cast<Annotation>();
}




} // namespace marco::ast
