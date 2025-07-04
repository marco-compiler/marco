#ifndef EXTERNAL_FUNCTION_CALL_H
#define EXTERNAL_FUNCTION_CALL_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast {
class ComponentReference; 
class ExternalFunctionCall : public ASTNode {
public:

  explicit ExternalFunctionCall(SourceRange location); 

  ExternalFunctionCall(const ExternalFunctionCall &other); 

  ~ExternalFunctionCall() override; 

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::External_Function_Call; 
  }

  std::unique_ptr<ASTNode> clone() const override;
  llvm::json::Value toJSON() const override;

  void setName(llvm::StringRef name); 
  llvm::StringRef getName() const; 

  void setComponentReference(std::unique_ptr<ASTNode> node); 
  ComponentReference *getComponentReference(); 
  const ComponentReference *getComponentReference() const; 

  void setExpressions(llvm::ArrayRef<std::unique_ptr<ASTNode>> expressions); 
  llvm::ArrayRef<std::unique_ptr<ASTNode>> getExpressions() const; 
  
  bool hasComponentReference() const;

private:
  std::string name;
  std::unique_ptr<ASTNode> componentReference; 
  llvm::SmallVector<std::unique_ptr<ASTNode>> expressions; 
};
}// namespace marco::ast

#endif 
