#include "marco/AST/BaseModelica/ArrayDimension.h"
#include "marco/AST/BaseModelica/Expression.h"

using namespace ::marco::ast;
using namespace ::marco::ast::bmodelica;

namespace {
struct ArrayDimensionJsonVisitor {
  llvm::json::Value operator()(int64_t value) {
    return llvm::json::Value(value);
  }

  llvm::json::Value operator()(const std::unique_ptr<ASTNode> &value) {
    return value->toJSON();
  }
};
} // namespace

namespace marco::ast::bmodelica {
ArrayDimension::ArrayDimension(SourceRange location)
    : ASTNode(ASTNodeKind::ArrayDimension, std::move(location)) {}

ArrayDimension::ArrayDimension(const ArrayDimension &other) : ASTNode(other) {
  if (other.hasExpression()) {
    setSize(other.getExpression()->clone());
  } else {
    setSize(other.getNumericSize());
  }
}

ArrayDimension::~ArrayDimension() = default;

std::unique_ptr<ASTNode> ArrayDimension::clone() const {
  return std::make_unique<ArrayDimension>(*this);
}

llvm::json::Value ArrayDimension::toJSON() const {
  llvm::json::Object result;

  ArrayDimensionJsonVisitor visitor;
  result["size"] = std::visit(visitor, size);

  addNodeKindToJSON(*this, result);
  return result;
}

bool ArrayDimension::hasExpression() const {
  return std::holds_alternative<std::unique_ptr<ASTNode>>(size);
}

bool ArrayDimension::isDynamic() const {
  return hasExpression() || getNumericSize() == kDynamicSize;
}

long ArrayDimension::getNumericSize() const {
  assert(std::holds_alternative<int64_t>(size));
  return std::get<int64_t>(size);
}

Expression *ArrayDimension::getExpression() {
  assert(hasExpression() && "Expression not set");
  return std::get<std::unique_ptr<ASTNode>>(size)->cast<Expression>();
}

const Expression *ArrayDimension::getExpression() const {
  assert(hasExpression() && "Expression not set");
  return std::get<std::unique_ptr<ASTNode>>(size)->cast<Expression>();
}

void ArrayDimension::setSize(int64_t value) { size = value; }

void ArrayDimension::setSize(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  node->setParent(this);
  size = std::move(node);
}
} // namespace marco::ast::bmodelica
