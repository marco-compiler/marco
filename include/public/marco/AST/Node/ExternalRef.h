#ifndef EXTERNAL_REF_H
#define EXTERNAL_REF_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast {

class ExternalFunctionCall; 
class Annotation; 

class ExternalRef : public ASTNode {
public:
  explicit ExternalRef(SourceRange location);

  ~ExternalRef() override;

  std::unique_ptr<ASTNode> clone() const override;
  llvm::json::Value toJSON() const override;

  void setLanguageSpecification(llvm::StringRef languageSpecification); 
  llvm::StringRef getLanguageSpecification(); 

  void setExternalFunctionCall(std::unique_ptr<ASTNode> externalFunctionCall); 
  const ExternalFunctionCall *getExternalFunctionCall() const; 
  ExternalFunctionCall *getExternalFunctionCall(); 

  void setAnnotationClause(std::unique_ptr<ASTNode> annotationClause); 
  const Annotation *getAnnotationClause() const; 
  Annotation *getAnnotationClause(); 

private:
  std::string languageSpecification; 
  std::unique_ptr<ASTNode> externalFunctionCall; 
  std::unique_ptr<ASTNode> annotationClause; 
};
} // namespace marco::ast

#endif