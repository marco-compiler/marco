#include "marco/AST/Node/ASTNode.h"
#include <memory>

namespace marco::ast {

class ExternalFunctionCall : public ASTNode {
public:

  explicit ExternalFunctionCall(SourceRange location); 

  ~ExternalFunctionCall() override; 

  std::unique_ptr<ASTNode> clone() const override;
  llvm::json::Value toJSON() const override;

  void setName(llvm::StringRef name); 
  llvm::StringRef getName() const; 

  void setComponentReference(std::unique_ptr<ASTNode> node); 
  std::unique_ptr<ASTNode> getComponentReference() const; 

  void setExpressions(llvm::ArrayRef<std::unique_ptr<ASTNode>> expressions); 
  llvm::ArrayRef<std::unique_ptr<ASTNode>> getExpressions() const; 

private:
  std::string name; 
  std::unique_ptr<ASTNode> componentReference; 
  llvm::ArrayRef<std::unique_ptr<ASTNode>> expressions; 
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_ALGORITHM_H
