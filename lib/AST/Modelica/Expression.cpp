#include "marco/AST/Modelica/Expression.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
llvm::StringRef toString(UnaryOperator op) {
  switch (op) {
  case UnaryOperator::Plus:
    return "+";
  case UnaryOperator::Minus:
    return "-";
  case UnaryOperator::PlusEW:
    return ".+";
  case UnaryOperator::MinusEW:
    return ".-";
  case UnaryOperator::Not:
    return "not";
  }

  llvm_unreachable("Unknown unary operator");
  return "";
}

llvm::StringRef toString(BinaryOperator op) {
  switch (op) {
  case BinaryOperator::Add:
    return "+";
  case BinaryOperator::AddEW:
    return ".+";
  case BinaryOperator::Sub:
    return "-";
  case BinaryOperator::SubEW:
    return ".-";
  case BinaryOperator::Mul:
    return "*";
  case BinaryOperator::MulEW:
    return ".*";
  case BinaryOperator::Div:
    return "/";
  case BinaryOperator::DivEW:
    return "./";
  case BinaryOperator::Pow:
    return "^";
  case BinaryOperator::PowEW:
    return ".^";
  case BinaryOperator::Equal:
    return "==";
  case BinaryOperator::NotEqual:
    return "<>";
  case BinaryOperator::Less:
    return "<";
  case BinaryOperator::LessEqual:
    return "<=";
  case BinaryOperator::Greater:
    return ">";
  case BinaryOperator::GreaterEqual:
    return ">=";
  case BinaryOperator::And:
    return "and";
  case BinaryOperator::Or:
    return "or";
  }

  llvm_unreachable("Unknown binary operator");
  return "";
}

//===---------------------------------------------------------------------===//
// Literals
//===---------------------------------------------------------------------===//

void IntegerLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["value"] = value;

  if (unit) {
    obj["unit"] = *unit;
  }
}

void RealLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["value"] = value;

  if (unit) {
    obj["unit"] = *unit;
  }
}

void StringLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["value"] = value;
}

void BooleanLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["value"] = value;
}

//===---------------------------------------------------------------------===//
// ComponentReference
//===---------------------------------------------------------------------===//

Name ComponentReference::getName() const {
  Name result;
  result.global = global;
  result.location = getLocation();

  for (const Part &part : parts) {
    result.parts.push_back(part.name);
  }

  return result;
}

void ComponentReference::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  for (const Part &part : parts) {
    walk(part.subscripts, callback);
  }
}

void ComponentReference::addJSONProperties(llvm::json::Object &obj) const {
  obj["global"] = global;
  llvm::json::Array array;

  for (const Part &part : parts) {
    llvm::json::Object partObj;
    partObj["name"] = part.name;

    if (!part.subscripts.empty()) {
      partObj["subscripts"] = modelica::toJSON(part.subscripts);
    }

    array.push_back(std::move(partObj));
  }

  obj["parts"] = std::move(array);
}

//===---------------------------------------------------------------------===//
// Operations
//===---------------------------------------------------------------------===//

void UnaryOperation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(operand, callback);
}

void UnaryOperation::addJSONProperties(llvm::json::Object &obj) const {
  obj["operator"] = toString(op);
  obj["operand"] = modelica::toJSON(operand.get());
}

void BinaryOperation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(lhs, callback);
  walk(rhs, callback);
}

void BinaryOperation::addJSONProperties(llvm::json::Object &obj) const {
  obj["operator"] = toString(op);
  obj["lhs"] = modelica::toJSON(lhs.get());
  obj["rhs"] = modelica::toJSON(rhs.get());
}

void Range::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(start, callback);
  walk(step, callback);
  walk(stop, callback);
}

void Range::addJSONProperties(llvm::json::Object &obj) const {
  obj["start"] = modelica::toJSON(start.get());

  if (step) {
    obj["step"] = modelica::toJSON(step.get());
  }

  obj["stop"] = modelica::toJSON(stop.get());
}

void IfExpression::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);
  walk(values, callback);
  walk(elseValue, callback);
}

void IfExpression::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["values"] = modelica::toJSON(values);
  obj["else"] = modelica::toJSON(elseValue.get());
}

//===---------------------------------------------------------------------===//
// Function calls and array constructors
//===---------------------------------------------------------------------===//

void Call::addNamedArgument(std::unique_ptr<NamedArgument> node) {
  namedArguments.emplace_back(std::move(node));
}

void Call::addIterator(std::unique_ptr<ForIndex> node) {
  iterators.emplace_back(std::move(node));
}

void Call::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(callee, callback);
  walk(arguments, callback);
  walk(namedArguments, callback);
  walk(iterators, callback);
}

void Call::addJSONProperties(llvm::json::Object &obj) const {
  obj["callee"] = modelica::toJSON(callee.get());
  obj["arguments"] = modelica::toJSON(arguments);

  if (!namedArguments.empty()) {
    obj["named_arguments"] = modelica::toJSON(namedArguments);
  }

  if (!iterators.empty()) {
    obj["iterators"] = modelica::toJSON(iterators);
  }
}

void FunctionPartialApplication::addNamedArgument(
    std::unique_ptr<NamedArgument> node) {
  namedArguments.emplace_back(std::move(node));
}

void FunctionPartialApplication::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(namedArguments, callback);
}

void FunctionPartialApplication::addJSONProperties(
    llvm::json::Object &obj) const {
  obj["function"] = modelica::toJSON(function);
  obj["named_arguments"] = modelica::toJSON(namedArguments);
}

void ArrayConstructor::addIterator(std::unique_ptr<ForIndex> node) {
  iterators.emplace_back(std::move(node));
}

void ArrayConstructor::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(elements, callback);
  walk(iterators, callback);
}

void ArrayConstructor::addJSONProperties(llvm::json::Object &obj) const {
  obj["elements"] = modelica::toJSON(elements);

  if (!iterators.empty()) {
    obj["iterators"] = modelica::toJSON(iterators);
  }
}

void ArrayConcatenation::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  for (const Children<Expression> &row : rows) {
    walk(row, callback);
  }
}

void ArrayConcatenation::addJSONProperties(llvm::json::Object &obj) const {
  llvm::json::Array array;

  for (const Children<Expression> &row : rows) {
    array.push_back(modelica::toJSON(row));
  }

  obj["rows"] = std::move(array);
}

void OutputList::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(elements, callback);
}

void OutputList::addJSONProperties(llvm::json::Object &obj) const {
  obj["elements"] = modelica::toJSON(elements);
}

void Subscripted::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(base, callback);
  walk(subscripts, callback);
}

void Subscripted::addJSONProperties(llvm::json::Object &obj) const {
  obj["base"] = modelica::toJSON(base.get());
  obj["subscripts"] = modelica::toJSON(subscripts);
}

void MemberAccess::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(base, callback);
}

void MemberAccess::addJSONProperties(llvm::json::Object &obj) const {
  obj["base"] = modelica::toJSON(base.get());
  obj["member"] = member;
}

//===---------------------------------------------------------------------===//
// Auxiliary nodes
//===---------------------------------------------------------------------===//

void ForIndex::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(range, callback);
}

void ForIndex::addJSONProperties(llvm::json::Object &obj) const {
  obj["name"] = name;
  obj["range"] = modelica::toJSON(range.get());
}

void NamedArgument::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(value, callback);
}

void NamedArgument::addJSONProperties(llvm::json::Object &obj) const {
  obj["name"] = name;
  obj["value"] = modelica::toJSON(value.get());
}
} // namespace marco::ast::modelica
