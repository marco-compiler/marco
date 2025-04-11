#include "marco/AST/Node/ASTNode.h"
#include <memory>

namespace marco::ast {

class ExternalRef : public ASTNode {
public:
  explicit ExternalRef(SourceRange location);

  ExternalRef(const ExternalRef &other);


  ~ExternalRef() override;

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;


private:
  std::string languageSpecification; 
  std::unique_ptr<ASTNode> externalFunctionCall; 
  std::unique_ptr<ASTNode> annotationClause; 
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_ALGORITHM_H
