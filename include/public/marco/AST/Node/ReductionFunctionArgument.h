#ifndef PUBLIC_MARCO_AST_NODE_REDUCTIONFUNCTIONARGUMENT_H
#define PUBLIC_MARCO_AST_NODE_REDUCTIONFUNCTIONARGUMENT_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/FunctionArgument.h"
#include "marco/Parser/Location.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <cstddef>
#include <memory>

namespace marco::ast {
class Expression;
class ForIndex;

class ReductionFunctionArgument : public FunctionArgument {
public:
  ReductionFunctionArgument(SourceRange location);

  ReductionFunctionArgument(const ReductionFunctionArgument &other);

  ~ReductionFunctionArgument() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::FunctionArgument_Reduction;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getExpression();

  const Expression *getExpression() const;

  void setExpression(std::unique_ptr<ASTNode> node);

  size_t getNumOfForIndices() const;

  ForIndex *getForIndex(size_t index);

  const ForIndex *getForIndex(size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getForIndices() const;

  void setForIndices(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  std::unique_ptr<ASTNode> expression;
  llvm::SmallVector<std::unique_ptr<ASTNode>> forIndices;
};
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_REDUCTIONFUNCTIONARGUMENT_H
