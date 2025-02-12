#include "marco/AST/Node/Reduction.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
Reduction::Reduction(SourceRange location)
    : Expression(ASTNode::Kind::Expression_Reduction, std::move(location)) {}

Reduction::Reduction(const Reduction &other) : Expression(other) {
  setCallee(other.callee->clone());
  setExpression(other.expression->clone());
  setIterators(other.iterators);
}

Reduction::~Reduction() = default;

std::unique_ptr<ASTNode> Reduction::clone() const {
  return std::make_unique<Reduction>(*this);
}

llvm::json::Value Reduction::toJSON() const {
  llvm::json::Object result;
  result["callee"] = getCallee()->toJSON();
  result["expression"] = getExpression()->toJSON();

  llvm::SmallVector<llvm::json::Value> iteratorsJson;

  for (const auto &iterator : iterators) {
    iteratorsJson.push_back(iterator->toJSON());
  }

  result["iterators"] = llvm::json::Array(iteratorsJson);

  addJSONProperties(result);
  return result;
}

bool Reduction::isLValue() const { return false; }

Expression *Reduction::getCallee() {
  assert(callee != nullptr && "Callee not set");
  return callee->cast<Expression>();
}

const Expression *Reduction::getCallee() const {
  assert(callee != nullptr && "Callee not set");
  return callee->cast<Expression>();
}

void Reduction::setCallee(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  callee = std::move(node);
  callee->setParent(this);
}

Expression *Reduction::getExpression() {
  assert(expression && "Expression not set");
  return expression->cast<Expression>();
}

const Expression *Reduction::getExpression() const {
  assert(expression && "Expression not set");
  return expression->cast<Expression>();
}

void Reduction::setExpression(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  expression = std::move(node);
  expression->setParent(this);
}

size_t Reduction::getNumOfIterators() const { return iterators.size(); }

Expression *Reduction::getIterator(size_t index) {
  assert(index < iterators.size());
  return iterators[index]->cast<Expression>();
}

const Expression *Reduction::getIterator(size_t index) const {
  assert(index < iterators.size());
  return iterators[index]->cast<Expression>();
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> Reduction::getIterators() const {
  return iterators;
}

void Reduction::setIterators(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  iterators.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Expression>());
    auto &clone = iterators.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast
