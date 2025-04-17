#include "marco/AST/Node/ASTNode.h"
#include <memory>

namespace marco::ast {

class ExternalRef : public ASTNode {
public:
  explicit ExternalRef(SourceRange location);

  ~ExternalRef() override;

  std::unique_ptr<ASTNode> clone() const override;
  llvm::json::Value toJSON() const override;

  void setLanguageSpecification(llvm::StringRef languageSpecification); 
  llvm::StringRef getLanguageSpecification(); 

  void setExternalFunctionCall(std::unique_ptr<ASTNode> externalFunctionCall); 
  std::unique_ptr<ASTNode> getExternalFunctionCall(); 

  void setAnnotationClause(std::unique_ptr<ASTNode> annotationClause); 
  std::unique_ptr<ASTNode> getAnnotationClause(); 


private:
  std::string languageSpecification; 
  std::unique_ptr<ASTNode> externalFunctionCall; 
  std::unique_ptr<ASTNode> annotationClause; 
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_ALGORITHM_H
