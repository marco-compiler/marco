#include "marco/AST/Node/IfEquation.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
IfEquation::IfEquation(SourceRange location)
    : Equation(ASTNode::Kind::Equation_If, std::move(location)) {}

IfEquation::IfEquation(const IfEquation &other) : Equation(other) {
  setIfCondition(other.ifCondition->clone());
  setIfEquations(other.ifEquations);

  setElseIfConditions(other.elseIfConditions);
  elseIfEquations.clear();
  elseIfEquations.resize(other.elseIfEquations.size());

  for (size_t i = 0, e = other.elseIfEquations.size(); i < e; ++i) {
    setElseIfEquations(i, other.elseIfEquations[i]);
  }

  setElseEquations(other.elseEquations);
}

IfEquation::~IfEquation() = default;

std::unique_ptr<ASTNode> IfEquation::clone() const {
  return std::make_unique<IfEquation>(*this);
}

llvm::json::Value IfEquation::toJSON() const {
  llvm::json::Object result;
  result["if_condition"] = getIfCondition()->toJSON();

  llvm::SmallVector<llvm::json::Value> ifEquationsJson;

  for (const auto &equation : ifEquations) {
    ifEquationsJson.push_back(equation->toJSON());
  }

  result["if_equations"] = llvm::json::Array(ifEquationsJson);
  llvm::SmallVector<llvm::json::Value> elseIfConditionsJson;

  for (const auto &condition : elseIfConditions) {
    elseIfConditionsJson.push_back(condition->toJSON());
  }

  llvm::SmallVector<llvm::json::Value> elseIfEquationsJson;

  for (const auto &equationsList : elseIfEquations) {
    llvm::SmallVector<llvm::json::Value> currentElseIfEquationsJson;

    for (const auto &equation : equationsList) {
      currentElseIfEquationsJson.push_back(equation->toJSON());
    }

    elseIfEquationsJson.push_back(
        llvm::json::Array(currentElseIfEquationsJson));
  }

  result["else_if_conditions"] = llvm::json::Array(elseIfConditionsJson);
  result["else_if_equations"] = llvm::json::Array(elseIfEquationsJson);

  llvm::SmallVector<llvm::json::Value> elseEquationsJson;

  for (const auto &equation : elseEquations) {
    elseEquationsJson.push_back(equation->toJSON());
  }

  result["else_equations"] = llvm::json::Array(elseEquationsJson);

  addJSONProperties(result);
  return result;
}

Expression *IfEquation::getIfCondition() {
  assert(ifCondition != nullptr && "If condition not set");
  return ifCondition->cast<Expression>();
}

const Expression *IfEquation::getIfCondition() const {
  assert(ifCondition != nullptr && "If condition not set");
  return ifCondition->cast<Expression>();
}

void IfEquation::setIfCondition(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  ifCondition = std::move(node);
  ifCondition->setParent(this);
}

size_t IfEquation::getNumOfIfEquations() const { return ifEquations.size(); }

Equation *IfEquation::getIfEquation(size_t index) {
  assert(index < ifEquations.size());
  return ifEquations[index]->cast<Equation>();
}

const Equation *IfEquation::getIfEquation(size_t index) const {
  assert(index < ifEquations.size());
  return ifEquations[index]->cast<Equation>();
}

void IfEquation::setIfEquations(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  ifEquations.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Equation>());
    auto &clone = ifEquations.emplace_back(node->clone());
    clone->setParent(this);
  }
}

size_t IfEquation::getNumOfElseIfConditions() const {
  return elseIfConditions.size();
}

Expression *IfEquation::getElseIfCondition(size_t index) {
  assert(index < elseIfConditions.size());
  return elseIfConditions[index]->cast<Expression>();
}

const Expression *IfEquation::getElseIfCondition(size_t index) const {
  assert(index < elseIfConditions.size());
  return elseIfConditions[index]->cast<Expression>();
}

void IfEquation::setElseIfConditions(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  elseIfConditions.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Expression>());
    auto &clone = elseIfConditions.emplace_back(node->clone());
    clone->setParent(this);
  }
}

size_t IfEquation::getNumOfElseIfEquations(size_t condition) const {
  assert(condition < elseIfEquations.size());
  return elseIfEquations[condition].size();
}

Equation *IfEquation::getElseIfEquation(size_t condition, size_t equation) {
  assert(equation < getNumOfElseIfEquations(condition));
  return elseIfEquations[condition][equation]->cast<Equation>();
}

const Equation *IfEquation::getElseIfEquation(size_t condition,
                                              size_t equation) const {
  assert(equation < getNumOfElseIfEquations(condition));
  return elseIfEquations[condition][equation]->cast<Equation>();
}

void IfEquation::setElseIfEquations(
    size_t condition, llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  if (condition > ifEquations.size()) {
    ifEquations.resize(condition + 1);
  }

  elseIfEquations[condition].clear();

  for (const auto &node : nodes) {
    assert(node->isa<Equation>());
    auto &clone = elseIfEquations[condition].emplace_back(node->clone());
    clone->setParent(this);
  }
}

size_t IfEquation::getNumOfElseEquations() const {
  return elseEquations.size();
}

bool IfEquation::hasElseEquations() const { return !elseEquations.empty(); }

Equation *IfEquation::getElseEquation(size_t index) {
  assert(index < elseEquations.size());
  return elseEquations[index]->cast<Equation>();
}

const Equation *IfEquation::getElseEquation(size_t index) const {
  assert(index < elseEquations.size());
  return elseEquations[index]->cast<Equation>();
}

void IfEquation::setElseEquations(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  elseEquations.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Equation>());
    auto &clone = elseEquations.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast
