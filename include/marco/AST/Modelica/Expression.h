#ifndef MARCO_AST_MODELICA_EXPRESSION_H
#define MARCO_AST_MODELICA_EXPRESSION_H

#include "marco/AST/Modelica/Node.h"

namespace marco::ast::modelica {
class ForIndex;
class NamedArgument;

/// Base class of the expressions.
class Expression : public Node {
public:
  using Node::Node;

  static bool classof(const ASTNode *node) {
    return node->getKind<NodeKind>() >= NodeKind::Expression &&
           node->getKind<NodeKind>() <= NodeKind::Expression_Last;
  }
};

//===---------------------------------------------------------------------===//
// Literals
//===---------------------------------------------------------------------===//

/// Integer literal, possibly with a unit.
class IntegerLiteral : public NodeImpl<IntegerLiteral, Expression,
                                       NodeKind::Expression_IntegerLiteral> {
public:
  using NodeImpl::NodeImpl;

  int64_t getValue() const { return value; }
  void setValue(int64_t newValue) { value = newValue; }

  const std::optional<std::string> &getUnit() const { return unit; }
  void setUnit(std::optional<std::string> newUnit) { unit = std::move(newUnit); }

  void walkChildren(llvm::function_ref<void(Node *)>) override {}

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  int64_t value = 0;
  std::optional<std::string> unit;
};

/// Real literal, possibly with a unit.
class RealLiteral
    : public NodeImpl<RealLiteral, Expression, NodeKind::Expression_RealLiteral> {
public:
  using NodeImpl::NodeImpl;

  double getValue() const { return value; }
  void setValue(double newValue) { value = newValue; }

  const std::optional<std::string> &getUnit() const { return unit; }
  void setUnit(std::optional<std::string> newUnit) { unit = std::move(newUnit); }

  void walkChildren(llvm::function_ref<void(Node *)>) override {}

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  double value = 0;
  std::optional<std::string> unit;
};

/// String literal, with the escape sequences already resolved.
class StringLiteral : public NodeImpl<StringLiteral, Expression,
                                      NodeKind::Expression_StringLiteral> {
public:
  using NodeImpl::NodeImpl;

  llvm::StringRef getValue() const { return value; }
  void setValue(std::string newValue) { value = std::move(newValue); }

  void walkChildren(llvm::function_ref<void(Node *)>) override {}

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  std::string value;
};

class BooleanLiteral : public NodeImpl<BooleanLiteral, Expression,
                                       NodeKind::Expression_BooleanLiteral> {
public:
  using NodeImpl::NodeImpl;

  bool getValue() const { return value; }
  void setValue(bool newValue) { value = newValue; }

  void walkChildren(llvm::function_ref<void(Node *)>) override {}

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  bool value = false;
};

/// The built-in variable `time`.
class Time : public NodeImpl<Time, Expression, NodeKind::Expression_Time> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override {}
};

/// The `end` keyword within array subscripts.
class End : public NodeImpl<End, Expression, NodeKind::Expression_End> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override {}
};

/// The `:` subscript, selecting a whole dimension.
class Colon : public NodeImpl<Colon, Expression, NodeKind::Expression_Colon> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)>) override {}
};

//===---------------------------------------------------------------------===//
// Component references
//===---------------------------------------------------------------------===//

/// Component reference (e.g., `a.b[1, :].c`). Each part has a name and
/// optional subscripts, which are expressions possibly including `Colon` and
/// `End` nodes.
class ComponentReference
    : public NodeImpl<ComponentReference, Expression,
                      NodeKind::Expression_ComponentReference> {
public:
  struct Part {
    std::string name;
    Children<Expression> subscripts;
    SourceRange location = SourceRange::unknown();
  };

  using NodeImpl::NodeImpl;

  bool isGlobal() const { return global; }
  void setGlobal(bool value) { global = value; }

  llvm::ArrayRef<Part> getParts() const { return parts; }
  llvm::MutableArrayRef<Part> getParts() { return parts; }
  void addPart(Part part) { parts.push_back(std::move(part)); }

  /// Get the name made of the parts, ignoring the subscripts.
  Name getName() const;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  bool global = false;
  std::vector<Part> parts;
};

//===---------------------------------------------------------------------===//
// Operations
//===---------------------------------------------------------------------===//

enum class UnaryOperator { Plus, Minus, PlusEW, MinusEW, Not };

enum class BinaryOperator {
  Add,
  AddEW,
  Sub,
  SubEW,
  Mul,
  MulEW,
  Div,
  DivEW,
  Pow,
  PowEW,
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  And,
  Or
};

llvm::StringRef toString(UnaryOperator op);
llvm::StringRef toString(BinaryOperator op);

class UnaryOperation : public NodeImpl<UnaryOperation, Expression,
                                       NodeKind::Expression_UnaryOperation> {
public:
  using NodeImpl::NodeImpl;

  UnaryOperator getOperator() const { return op; }
  void setOperator(UnaryOperator newOp) { op = newOp; }

  Expression *getOperand() const { return operand.get(); }
  void setOperand(std::unique_ptr<Expression> node) { operand = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  UnaryOperator op = UnaryOperator::Minus;
  Child<Expression> operand;
};

class BinaryOperation : public NodeImpl<BinaryOperation, Expression,
                                        NodeKind::Expression_BinaryOperation> {
public:
  using NodeImpl::NodeImpl;

  BinaryOperator getOperator() const { return op; }
  void setOperator(BinaryOperator newOp) { op = newOp; }

  Expression *getLhs() const { return lhs.get(); }
  void setLhs(std::unique_ptr<Expression> node) { lhs = std::move(node); }

  Expression *getRhs() const { return rhs.get(); }
  void setRhs(std::unique_ptr<Expression> node) { rhs = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  BinaryOperator op = BinaryOperator::Add;
  Child<Expression> lhs;
  Child<Expression> rhs;
};

/// Range expression `start:stop` or `start:step:stop`.
class Range : public NodeImpl<Range, Expression, NodeKind::Expression_Range> {
public:
  using NodeImpl::NodeImpl;

  Expression *getStart() const { return start.get(); }
  void setStart(std::unique_ptr<Expression> node) { start = std::move(node); }

  /// Get the step, if specified.
  Expression *getStep() const { return step.get(); }
  void setStep(std::unique_ptr<Expression> node) { step = std::move(node); }

  Expression *getStop() const { return stop.get(); }
  void setStop(std::unique_ptr<Expression> node) { stop = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<Expression> start;
  Child<Expression> step;
  Child<Expression> stop;
};

/// If-expression. The `elseif` branches are kept as
/// separate conditions.
class IfExpression
    : public NodeImpl<IfExpression, Expression, NodeKind::Expression_If> {
public:
  using NodeImpl::NodeImpl;

  /// Conditions of the if and elseif branches.
  const Children<Expression> &getConditions() const { return conditions; }

  /// Values of the if and elseif branches.
  const Children<Expression> &getValues() const { return values; }

  void addBranch(std::unique_ptr<Expression> condition,
                 std::unique_ptr<Expression> value) {
    conditions.emplace_back(std::move(condition));
    values.emplace_back(std::move(value));
  }

  Expression *getElseValue() const { return elseValue.get(); }
  void setElseValue(std::unique_ptr<Expression> node) {
    elseValue = std::move(node);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<Expression> conditions;
  Children<Expression> values;
  Child<Expression> elseValue;
};

//===---------------------------------------------------------------------===//
// Function calls and array constructors
//===---------------------------------------------------------------------===//

/// Function call. The callee is a component reference,
/// whose single part is `der`, `initial` or `pure` for the homonymous
/// operators. The arguments are either positional and named arguments, or a
/// single expression followed by iterators (e.g., `sum(i for i in 1:n)`).
class Call : public NodeImpl<Call, Expression, NodeKind::Expression_Call> {
public:
  using NodeImpl::NodeImpl;

  ComponentReference *getCallee() const { return callee.get(); }
  void setCallee(std::unique_ptr<ComponentReference> node) {
    callee = std::move(node);
  }

  const Children<Expression> &getArguments() const { return arguments; }
  void addArgument(std::unique_ptr<Expression> node) {
    arguments.emplace_back(std::move(node));
  }

  const Children<NamedArgument> &getNamedArguments() const {
    return namedArguments;
  }

  void addNamedArgument(std::unique_ptr<NamedArgument> node);

  /// Iterators of reductions and array constructors with iterators. If
  /// present, the call has exactly one positional argument.
  const Children<ForIndex> &getIterators() const { return iterators; }
  void addIterator(std::unique_ptr<ForIndex> node);

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<ComponentReference> callee;
  Children<Expression> arguments;
  Children<NamedArgument> namedArguments;
  Children<ForIndex> iterators;
};

/// Function partial application, usable only as
/// function argument.
class FunctionPartialApplication
    : public NodeImpl<FunctionPartialApplication, Expression,
                      NodeKind::Expression_FunctionPartialApplication> {
public:
  using NodeImpl::NodeImpl;

  const Name &getFunction() const { return function; }
  void setFunction(Name name) { function = std::move(name); }

  const Children<NamedArgument> &getNamedArguments() const {
    return namedArguments;
  }

  void addNamedArgument(std::unique_ptr<NamedArgument> node);

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Name function;
  Children<NamedArgument> namedArguments;
};

/// Array constructor `{a, b, c}` or with iterators `{e for i in r}`.
class ArrayConstructor
    : public NodeImpl<ArrayConstructor, Expression,
                      NodeKind::Expression_ArrayConstructor> {
public:
  using NodeImpl::NodeImpl;

  const Children<Expression> &getElements() const { return elements; }
  void addElement(std::unique_ptr<Expression> node) {
    elements.emplace_back(std::move(node));
  }

  /// Iterators of the array constructor. If present, the constructor has
  /// exactly one element.
  const Children<ForIndex> &getIterators() const { return iterators; }
  void addIterator(std::unique_ptr<ForIndex> node);

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<Expression> elements;
  Children<ForIndex> iterators;
};

/// Array concatenation `[a, b; c, d]`.
class ArrayConcatenation
    : public NodeImpl<ArrayConcatenation, Expression,
                      NodeKind::Expression_ArrayConcatenation> {
public:
  using NodeImpl::NodeImpl;

  const std::vector<Children<Expression>> &getRows() const { return rows; }
  void addRow(Children<Expression> row) { rows.push_back(std::move(row)); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  std::vector<Children<Expression>> rows;
};

/// Parenthesized list of expressions with possibly omitted elements, as in
/// `(a, , b) = f(x)`. A single parenthesized expression is not represented
/// by this node.
class OutputList
    : public NodeImpl<OutputList, Expression, NodeKind::Expression_OutputList> {
public:
  using NodeImpl::NodeImpl;

  /// Get the elements. Omitted elements are null.
  const Children<Expression> &getElements() const { return elements; }
  void addElement(std::unique_ptr<Expression> node) {
    elements.emplace_back(std::move(node));
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<Expression> elements;
};

/// Subscripts applied to a parenthesized expression, as in `(a * b)[1, :]`.
class Subscripted : public NodeImpl<Subscripted, Expression,
                                    NodeKind::Expression_Subscripted> {
public:
  using NodeImpl::NodeImpl;

  Expression *getBase() const { return base.get(); }
  void setBase(std::unique_ptr<Expression> node) { base = std::move(node); }

  const Children<Expression> &getSubscripts() const { return subscripts; }
  void addSubscript(std::unique_ptr<Expression> node) {
    subscripts.emplace_back(std::move(node));
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<Expression> base;
  Children<Expression> subscripts;
};

/// Member access applied to a parenthesized expression, as in `(f(x)).a`.
class MemberAccess : public NodeImpl<MemberAccess, Expression,
                                     NodeKind::Expression_MemberAccess> {
public:
  using NodeImpl::NodeImpl;

  Expression *getBase() const { return base.get(); }
  void setBase(std::unique_ptr<Expression> node) { base = std::move(node); }

  llvm::StringRef getMember() const { return member; }
  void setMember(std::string name) { member = std::move(name); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<Expression> base;
  std::string member;
};

//===---------------------------------------------------------------------===//
// Auxiliary nodes
//===---------------------------------------------------------------------===//

/// Iterator of for-loops, reductions and array constructors
/// (`i in 1:n`). The range is null if implicit.
class ForIndex : public NodeImpl<ForIndex, Node, NodeKind::ForIndex> {
public:
  using NodeImpl::NodeImpl;

  llvm::StringRef getName() const { return name; }
  void setName(std::string newName) { name = std::move(newName); }

  Expression *getRange() const { return range.get(); }
  void setRange(std::unique_ptr<Expression> node) { range = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  std::string name;
  Child<Expression> range;
};

/// Named argument of a function call (`name = value`).
class NamedArgument
    : public NodeImpl<NamedArgument, Node, NodeKind::NamedArgument> {
public:
  using NodeImpl::NodeImpl;

  llvm::StringRef getName() const { return name; }
  void setName(std::string newName) { name = std::move(newName); }

  Expression *getValue() const { return value.get(); }
  void setValue(std::unique_ptr<Expression> node) { value = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  std::string name;
  Child<Expression> value;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_EXPRESSION_H
