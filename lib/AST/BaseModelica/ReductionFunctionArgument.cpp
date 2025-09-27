#include "marco/AST/BaseModelica/ReductionFunctionArgument.h"
#include "marco/AST/BaseModelica/Expression.h"
#include "marco/AST/BaseModelica/ForIndex.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
ReductionFunctionArgument::ReductionFunctionArgument(SourceRange location)
    : FunctionArgument(ASTNodeKind::FunctionArgument_Reduction,
                       std::move(location)) {}

ReductionFunctionArgument::ReductionFunctionArgument(
    const ReductionFunctionArgument &other)
    : FunctionArgument(other) {
  setExpression(other.expression->clone());
  setForIndices(other.forIndices);
}

ReductionFunctionArgument::~ReductionFunctionArgument() = default;

std::unique_ptr<ASTNode> ReductionFunctionArgument::clone() const {
  return std::make_unique<ReductionFunctionArgument>(*this);
}

llvm::json::Value ReductionFunctionArgument::toJSON() const {
  llvm::json::Object result;
  result["expression"] = getExpression()->toJSON();

  llvm::SmallVector<llvm::json::Value> iteratorsJson;

  for (const auto &forIndex : forIndices) {
    iteratorsJson.push_back(forIndex->toJSON());
  }

  result["iterators"] = llvm::json::Array(iteratorsJson);

  addNodeKindToJSON(*this, result);
  return result;
}

Expression *ReductionFunctionArgument::getExpression() {
  assert(expression && "Expression not set");
  return expression->cast<Expression>();
}

const Expression *ReductionFunctionArgument::getExpression() const {
  assert(expression && "Expression not set");
  return expression->cast<Expression>();
}

void ReductionFunctionArgument::setExpression(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  expression = std::move(node);
  expression->setParent(this);
}

size_t ReductionFunctionArgument::getNumOfForIndices() const {
  return forIndices.size();
}

ForIndex *ReductionFunctionArgument::getForIndex(size_t index) {
  assert(index < forIndices.size());
  return forIndices[index]->cast<ForIndex>();
}

const ForIndex *ReductionFunctionArgument::getForIndex(size_t index) const {
  assert(index < forIndices.size());
  return forIndices[index]->cast<ForIndex>();
}

llvm::ArrayRef<std::unique_ptr<ASTNode>>
ReductionFunctionArgument::getForIndices() const {
  return forIndices;
}

void ReductionFunctionArgument::setForIndices(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  forIndices.clear();

  for (const auto &node : nodes) {
    assert(node->isa<ForIndex>());
    auto &clone = forIndices.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast::bmodelica
