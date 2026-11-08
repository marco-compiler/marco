#include "marco/AST/Modelica/Equation.h"

using namespace ::marco::ast::modelica;

namespace {
template <typename T>
llvm::json::Value branchesToJSON(const std::vector<Children<T>> &branches) {
  llvm::json::Array array;

  for (const Children<T> &branch : branches) {
    array.push_back(toJSON(branch));
  }

  return array;
}
} // namespace

namespace marco::ast::modelica {
//===---------------------------------------------------------------------===//
// Equations
//===---------------------------------------------------------------------===//

void Equation::addJSONProperties(llvm::json::Object &obj) const {
  description.addToJSON(obj);
}

void EqualityEquation::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(lhs, callback);
  walk(rhs, callback);
  description.walkChildren(callback);
}

void EqualityEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["lhs"] = modelica::toJSON(lhs.get());
  obj["rhs"] = modelica::toJSON(rhs.get());
  Equation::addJSONProperties(obj);
}

void CallEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(call, callback);
  description.walkChildren(callback);
}

void CallEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["call"] = modelica::toJSON(call.get());
  Equation::addJSONProperties(obj);
}

void ConnectEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(lhs, callback);
  walk(rhs, callback);
  description.walkChildren(callback);
}

void ConnectEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["lhs"] = modelica::toJSON(lhs.get());
  obj["rhs"] = modelica::toJSON(rhs.get());
  Equation::addJSONProperties(obj);
}

void IfEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);

  for (const Children<Equation> &branch : branches) {
    walk(branch, callback);
  }

  walk(elseEquations, callback);
  description.walkChildren(callback);
}

void IfEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["branches"] = branchesToJSON(branches);
  obj["else"] = modelica::toJSON(elseEquations);
  Equation::addJSONProperties(obj);
}

void ForEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(indices, callback);
  walk(equations, callback);
  description.walkChildren(callback);
}

void ForEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["indices"] = modelica::toJSON(indices);
  obj["equations"] = modelica::toJSON(equations);
  Equation::addJSONProperties(obj);
}

void WhenEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);

  for (const Children<Equation> &branch : branches) {
    walk(branch, callback);
  }

  description.walkChildren(callback);
}

void WhenEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["branches"] = branchesToJSON(branches);
  Equation::addJSONProperties(obj);
}

//===---------------------------------------------------------------------===//
// Statements
//===---------------------------------------------------------------------===//

void Statement::addJSONProperties(llvm::json::Object &obj) const {
  description.addToJSON(obj);
}

void AssignmentStatement::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(destination, callback);
  walk(value, callback);
  description.walkChildren(callback);
}

void AssignmentStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["destination"] = modelica::toJSON(destination.get());
  obj["value"] = modelica::toJSON(value.get());
  Statement::addJSONProperties(obj);
}

void MultipleAssignmentStatement::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(destinations, callback);
  walk(call, callback);
  description.walkChildren(callback);
}

void MultipleAssignmentStatement::addJSONProperties(
    llvm::json::Object &obj) const {
  obj["destinations"] = modelica::toJSON(destinations);
  obj["call"] = modelica::toJSON(call.get());
  Statement::addJSONProperties(obj);
}

void CallStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(call, callback);
  description.walkChildren(callback);
}

void CallStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["call"] = modelica::toJSON(call.get());
  Statement::addJSONProperties(obj);
}

void BreakStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  description.walkChildren(callback);
}

void ReturnStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  description.walkChildren(callback);
}

void IfStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);

  for (const Children<Statement> &branch : branches) {
    walk(branch, callback);
  }

  walk(elseStatements, callback);
  description.walkChildren(callback);
}

void IfStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["branches"] = branchesToJSON(branches);
  obj["else"] = modelica::toJSON(elseStatements);
  Statement::addJSONProperties(obj);
}

void ForStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(indices, callback);
  walk(statements, callback);
  description.walkChildren(callback);
}

void ForStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["indices"] = modelica::toJSON(indices);
  obj["statements"] = modelica::toJSON(statements);
  Statement::addJSONProperties(obj);
}

void WhileStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(condition, callback);
  walk(statements, callback);
  description.walkChildren(callback);
}

void WhileStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["condition"] = modelica::toJSON(condition.get());
  obj["statements"] = modelica::toJSON(statements);
  Statement::addJSONProperties(obj);
}

void WhenStatement::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);

  for (const Children<Statement> &branch : branches) {
    walk(branch, callback);
  }

  description.walkChildren(callback);
}

void WhenStatement::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["branches"] = branchesToJSON(branches);
  Statement::addJSONProperties(obj);
}
} // namespace marco::ast::modelica
