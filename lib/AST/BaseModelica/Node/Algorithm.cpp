#include "marco/AST/BaseModelica/Node/Algorithm.h"
#include "marco/AST/BaseModelica/Node/Statement.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
Algorithm::Algorithm(SourceRange location)
    : ASTNode(ASTNode::Kind::Algorithm, std::move(location)) {}

Algorithm::Algorithm(const Algorithm &other)
    : ASTNode(other), initial(other.initial) {
  setStatements(other.statements);
}

Algorithm::~Algorithm() = default;

std::unique_ptr<ASTNode> Algorithm::clone() const {
  return std::make_unique<Algorithm>(*this);
}

llvm::json::Value Algorithm::toJSON() const {
  llvm::json::Object result;

  result["initial"] = initial;
  llvm::SmallVector<llvm::json::Value> statementsJson;

  for (const auto &statement : statements) {
    statementsJson.push_back(statement->toJSON());
  }

  result["statements"] = llvm::json::Array(statementsJson);

  addJSONProperties(result);
  return result;
}

bool Algorithm::isInitial() const { return initial; }

void Algorithm::setInitial(bool value) { initial = value; }

size_t Algorithm::size() const { return statements.size(); }

bool Algorithm::empty() const { return statements.empty(); }

Statement *Algorithm::operator[](size_t index) {
  assert(index < statements.size());
  return statements[index]->cast<Statement>();
}

const Statement *Algorithm::operator[](size_t index) const {
  assert(index < statements.size());
  return statements[index]->cast<Statement>();
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> Algorithm::getStatements() {
  return statements;
}

void Algorithm::setStatements(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  statements.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Statement>());
    auto &clone = statements.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast::bmodelica
