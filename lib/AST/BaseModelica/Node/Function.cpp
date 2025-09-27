#include "marco/AST/BaseModelica/Node/Function.h"
#include "marco/AST/BaseModelica/Node/Annotation.h"
#include "marco/AST/BaseModelica/Node/Expression.h"
#include "marco/AST/BaseModelica/Node/Member.h"
#include "marco/AST/BaseModelica/Node/Type.h"
#include <algorithm>
#include <variant>

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
FunctionType::FunctionType(llvm::ArrayRef<std::unique_ptr<ASTNode>> args,
                           llvm::ArrayRef<std::unique_ptr<ASTNode>> results) {
  for (const auto &arg : args) {
    this->args.push_back(arg->clone());
  }

  for (const auto &result : results) {
    this->results.push_back(result->clone());
  }
}

size_t FunctionType::getNumOfArgs() const { return args.size(); }

const VariableType *FunctionType::getArg(size_t index) const {
  assert(index < args.size());
  return args[index]->cast<VariableType>();
}

size_t FunctionType::getNumOfResults() const { return results.size(); }

const VariableType *FunctionType::getResult(size_t index) const {
  assert(index < results.size());
  return results[index]->cast<VariableType>();
}

PartialDerFunction::PartialDerFunction(SourceRange location)
    : Function(ASTNode::Kind::Class_Function_PartialDerFunction,
               std::move(location)) {}

PartialDerFunction::PartialDerFunction(const PartialDerFunction &other)
    : Function(other) {
  setDerivedFunction(other.derivedFunction->clone());
  setIndependentVariables(other.independentVariables);
}

PartialDerFunction::~PartialDerFunction() = default;

std::unique_ptr<ASTNode> PartialDerFunction::clone() const {
  return std::make_unique<PartialDerFunction>(*this);
}

llvm::json::Value PartialDerFunction::toJSON() const {
  llvm::json::Object result;
  result["derived_function"] = getDerivedFunction()->toJSON();

  llvm::SmallVector<llvm::json::Value> independentVariablesJson;

  for (const auto &independentVariable : independentVariables) {
    independentVariablesJson.push_back(independentVariable->toJSON());
  }

  result["independent_variables"] = llvm::json::Array(independentVariablesJson);

  addJSONProperties(result);
  return result;
}

Expression *PartialDerFunction::getDerivedFunction() const {
  assert(derivedFunction != nullptr && "Derived function not set");
  return derivedFunction->cast<Expression>();
}

void PartialDerFunction::setDerivedFunction(std::unique_ptr<ASTNode> node) {
  derivedFunction = std::move(node);
  derivedFunction->setParent(this);
}

llvm::ArrayRef<std::unique_ptr<ASTNode>>
PartialDerFunction::getIndependentVariables() const {
  return independentVariables;
}

void PartialDerFunction::setIndependentVariables(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  independentVariables.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Expression>());
    auto &clone = independentVariables.emplace_back(node->clone());
    clone->setParent(this);
  }
}

StandardFunction::StandardFunction(SourceRange location)
    : Function(ASTNode::Kind::Class_Function_StandardFunction,
               std::move(location)) {}

StandardFunction::StandardFunction(const StandardFunction &other) = default;

StandardFunction::~StandardFunction() = default;

std::unique_ptr<ASTNode> StandardFunction::clone() const {
  return std::make_unique<StandardFunction>(*this);
}

llvm::json::Value StandardFunction::toJSON() const {
  llvm::json::Object result;
  result["pure"] = llvm::json::Value(pure);

  if (hasAnnotation()) {
    result["annotation"] = getAnnotation()->toJSON();
  }

  addJSONProperties(result);
  return result;
}

bool StandardFunction::isPure() const { return pure; }

void StandardFunction::setPure(bool value) { pure = value; }

bool StandardFunction::shouldBeInlined() const {
  return hasAnnotation() && getAnnotation()->getInlineProperty();
}

FunctionType StandardFunction::getType() const {
  llvm::SmallVector<std::unique_ptr<ASTNode>> args;
  llvm::SmallVector<std::unique_ptr<ASTNode>> results;

  for (const auto &node : getVariables()) {
    auto *variable = node->cast<Member>();

    if (variable->isInput()) {
      args.push_back(variable->getType()->clone());
    } else if (variable->isOutput()) {
      results.push_back(variable->getType()->clone());
    }
  }

  return FunctionType(args, results);
}

DerivativeAnnotation::DerivativeAnnotation(llvm::StringRef name,
                                           unsigned int order)
    : name(name.str()), order(order) {
  assert(order > 0);
}

llvm::StringRef DerivativeAnnotation::getName() const { return name; }

unsigned int DerivativeAnnotation::getOrder() const { return order; }

InverseFunctionAnnotation::InverseFunctionAnnotation() = default;

bool InverseFunctionAnnotation::isInvertible(llvm::StringRef arg) const {
  return map.find(arg) != map.end();
}

llvm::StringRef InverseFunctionAnnotation::getInverseFunction(
    llvm::StringRef invertibleArg) const {
  assert(isInvertible(invertibleArg));
  return map.find(invertibleArg)->second.first;
}

llvm::ArrayRef<std::string>
InverseFunctionAnnotation::getInverseArgs(llvm::StringRef invertibleArg) const {
  assert(isInvertible(invertibleArg));
  return map.find(invertibleArg)->second.second;
}

void InverseFunctionAnnotation::addInverse(llvm::StringRef invertedArg,
                                           llvm::StringRef inverseFunctionName,
                                           llvm::ArrayRef<std::string> args) {
  assert(map.find(invertedArg) == map.end());
  Container<std::string> c(args.begin(), args.end());
  map[invertedArg] = std::make_pair(inverseFunctionName.str(), std::move(c));
}
} // namespace marco::ast::bmodelica
