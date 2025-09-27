#include "marco/AST/BaseModelica/Node/WhenEquation.h"
#include "marco/AST/BaseModelica/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
WhenEquation::WhenEquation(SourceRange location)
    : Equation(ASTNode::Kind::Equation_If, std::move(location)) {}

WhenEquation::WhenEquation(const WhenEquation &other) : Equation(other) {
  setWhenCondition(other.whenCondition->clone());
  setWhenEquations(other.whenEquations);

  setElseWhenConditions(other.elseWhenConditions);
  elseWhenEquations.clear();
  elseWhenEquations.resize(other.elseWhenEquations.size());

  for (size_t i = 0, e = other.elseWhenEquations.size(); i < e; ++i) {
    setElseWhenEquations(i, other.elseWhenEquations[i]);
  }

  setElseEquations(other.elseEquations);
}

WhenEquation::~WhenEquation() = default;

std::unique_ptr<ASTNode> WhenEquation::clone() const {
  return std::make_unique<WhenEquation>(*this);
}

llvm::json::Value WhenEquation::toJSON() const {
  llvm::json::Object result;
  result["when_condition"] = getWhenCondition()->toJSON();

  llvm::SmallVector<llvm::json::Value> ifEquationsJson;

  for (const auto &equation : whenEquations) {
    ifEquationsJson.push_back(equation->toJSON());
  }

  result["when_equations"] = llvm::json::Array(ifEquationsJson);
  llvm::SmallVector<llvm::json::Value> elseIfConditionsJson;

  for (const auto &condition : elseWhenConditions) {
    elseIfConditionsJson.push_back(condition->toJSON());
  }

  llvm::SmallVector<llvm::json::Value> elseIfEquationsJson;

  for (const auto &equationsList : elseWhenEquations) {
    llvm::SmallVector<llvm::json::Value> currentElseWhenEquationsJson;

    for (const auto &equation : equationsList) {
      currentElseWhenEquationsJson.push_back(equation->toJSON());
    }

    elseIfEquationsJson.push_back(
        llvm::json::Array(currentElseWhenEquationsJson));
  }

  result["else_when_conditions"] = llvm::json::Array(elseIfConditionsJson);
  result["else_when_equations"] = llvm::json::Array(elseIfEquationsJson);

  llvm::SmallVector<llvm::json::Value> elseEquationsJson;

  for (const auto &equation : elseEquations) {
    elseEquationsJson.push_back(equation->toJSON());
  }

  result["else_equations"] = llvm::json::Array(elseEquationsJson);

  addJSONProperties(result);
  return result;
}

Expression *WhenEquation::getWhenCondition() {
  assert(whenCondition != nullptr && "When condition not set");
  return whenCondition->cast<Expression>();
}

const Expression *WhenEquation::getWhenCondition() const {
  assert(whenCondition != nullptr && "When condition not set");
  return whenCondition->cast<Expression>();
}

void WhenEquation::setWhenCondition(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  whenCondition = std::move(node);
  whenCondition->setParent(this);
}

size_t WhenEquation::getNumOfWhenEquations() const {
  return whenEquations.size();
}

Equation *WhenEquation::getWhenEquation(size_t index) {
  assert(index < whenEquations.size());
  return whenEquations[index]->cast<Equation>();
}

const Equation *WhenEquation::getWhenEquation(size_t index) const {
  assert(index < whenEquations.size());
  return whenEquations[index]->cast<Equation>();
}

void WhenEquation::setWhenEquations(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  whenEquations.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Equation>());
    auto &clone = whenEquations.emplace_back(node->clone());
    clone->setParent(this);
  }
}

size_t WhenEquation::getNumOfElseWhenConditions() const {
  return elseWhenConditions.size();
}

Expression *WhenEquation::getElseWhenCondition(size_t index) {
  assert(index < elseWhenConditions.size());
  return elseWhenConditions[index]->cast<Expression>();
}

const Expression *WhenEquation::getElseWhenCondition(size_t index) const {
  assert(index < elseWhenConditions.size());
  return elseWhenConditions[index]->cast<Expression>();
}

void WhenEquation::setElseWhenConditions(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  elseWhenConditions.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Expression>());
    auto &clone = elseWhenConditions.emplace_back(node->clone());
    clone->setParent(this);
  }
}

size_t WhenEquation::getNumOfElseWhenEquations(size_t condition) const {
  assert(condition < elseWhenEquations.size());
  return elseWhenEquations[condition].size();
}

Equation *WhenEquation::getElseWhenEquation(size_t condition, size_t equation) {
  assert(equation < getNumOfElseWhenEquations(condition));
  return elseWhenEquations[condition][equation]->cast<Equation>();
}

const Equation *WhenEquation::getElseWhenEquation(size_t condition,
                                                  size_t equation) const {
  assert(equation < getNumOfElseWhenEquations(condition));
  return elseWhenEquations[condition][equation]->cast<Equation>();
}

void WhenEquation::setElseWhenEquations(
    size_t condition, llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  if (condition > whenEquations.size()) {
    whenEquations.resize(condition + 1);
  }

  elseWhenEquations[condition].clear();

  for (const auto &node : nodes) {
    assert(node->isa<Equation>());
    auto &clone = elseWhenEquations[condition].emplace_back(node->clone());
    clone->setParent(this);
  }
}

size_t WhenEquation::getNumOfElseEquations() const {
  return elseEquations.size();
}

bool WhenEquation::hasElseEquations() const { return !elseEquations.empty(); }

Equation *WhenEquation::getElseEquation(size_t index) {
  assert(index < elseEquations.size());
  return elseEquations[index]->cast<Equation>();
}

const Equation *WhenEquation::getElseEquation(size_t index) const {
  assert(index < elseEquations.size());
  return elseEquations[index]->cast<Equation>();
}

void WhenEquation::setElseEquations(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  elseEquations.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Equation>());
    auto &clone = elseEquations.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast::bmodelica
