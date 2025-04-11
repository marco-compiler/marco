#include "marco/AST/Node/ASTNode.h"
#include <memory>

namespace marco::ast {

class ExternalFunctionCall : public ASTNode {
public:
  explicit ExternalFunctionCall(SourceRange location);

  ~ExternalFunctionCall() override;

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

private:
  std::unique_ptr<ASTNode> lhs; 
  std::unique_ptr<ASTNode> operation; 

};
} // namespace marco::ast

#endif // MARCO_AST_NODE_ALGORITHM_H
