#ifndef MARCO_AST_MODELICA_EQUATION_H
#define MARCO_AST_MODELICA_EQUATION_H

#include "marco/AST/Modelica/Modification.h"

namespace marco::ast::modelica {
//===---------------------------------------------------------------------===//
// Equations
//===---------------------------------------------------------------------===//

/// Base class of the equations.
class Equation : public Node {
public:
  using Node::Node;

  static bool classof(const ASTNode *node) {
    return node->getKind<NodeKind>() >= NodeKind::Equation &&
           node->getKind<NodeKind>() <= NodeKind::Equation_Last;
  }

  Description &getDescription() { return description; }
  const Description &getDescription() const { return description; }

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

  Description description;
};

/// Equation `lhs = rhs`. The left-hand side may be an
/// OutputList, as in `(a, b) = f(x)`.
class EqualityEquation : public NodeImpl<EqualityEquation, Equation,
                                         NodeKind::Equation_Equality> {
public:
  using NodeImpl::NodeImpl;

  Expression *getLhs() const { return lhs.get(); }
  void setLhs(std::unique_ptr<Expression> node) { lhs = std::move(node); }

  Expression *getRhs() const { return rhs.get(); }
  void setRhs(std::unique_ptr<Expression> node) { rhs = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<Expression> lhs;
  Child<Expression> rhs;
};

/// Function call used as equation, such as `assert(...)` or
/// `Connections.branch(...)`.
class CallEquation
    : public NodeImpl<CallEquation, Equation, NodeKind::Equation_Call> {
public:
  using NodeImpl::NodeImpl;

  Call *getCall() const { return call.get(); }
  void setCall(std::unique_ptr<Call> node) { call = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<Call> call;
};

/// Connect-equation.
class ConnectEquation
    : public NodeImpl<ConnectEquation, Equation, NodeKind::Equation_Connect> {
public:
  using NodeImpl::NodeImpl;

  ComponentReference *getLhs() const { return lhs.get(); }
  void setLhs(std::unique_ptr<ComponentReference> node) {
    lhs = std::move(node);
  }

  ComponentReference *getRhs() const { return rhs.get(); }
  void setRhs(std::unique_ptr<ComponentReference> node) {
    rhs = std::move(node);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<ComponentReference> lhs;
  Child<ComponentReference> rhs;
};

/// If-equation. Each branch has a condition, except for
/// the optional else branch.
class IfEquation : public NodeImpl<IfEquation, Equation, NodeKind::Equation_If> {
public:
  using NodeImpl::NodeImpl;

  const Children<Expression> &getConditions() const { return conditions; }
  const std::vector<Children<Equation>> &getBranches() const { return branches; }

  void addBranch(std::unique_ptr<Expression> condition,
                 Children<Equation> equations) {
    conditions.emplace_back(std::move(condition));
    branches.push_back(std::move(equations));
  }

  const Children<Equation> &getElseEquations() const { return elseEquations; }
  void setElseEquations(Children<Equation> equations) {
    elseEquations = std::move(equations);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<Expression> conditions;
  std::vector<Children<Equation>> branches;
  Children<Equation> elseEquations;
};

/// For-equation.
class ForEquation
    : public NodeImpl<ForEquation, Equation, NodeKind::Equation_For> {
public:
  using NodeImpl::NodeImpl;

  const Children<ForIndex> &getIndices() const { return indices; }
  void addIndex(std::unique_ptr<ForIndex> node) {
    indices.emplace_back(std::move(node));
  }

  const Children<Equation> &getEquations() const { return equations; }
  void setEquations(Children<Equation> body) { equations = std::move(body); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<ForIndex> indices;
  Children<Equation> equations;
};

/// When-equation. The first branch is the `when` one,
/// the others are the `elsewhen` ones.
class WhenEquation
    : public NodeImpl<WhenEquation, Equation, NodeKind::Equation_When> {
public:
  using NodeImpl::NodeImpl;

  const Children<Expression> &getConditions() const { return conditions; }
  const std::vector<Children<Equation>> &getBranches() const { return branches; }

  void addBranch(std::unique_ptr<Expression> condition,
                 Children<Equation> equations) {
    conditions.emplace_back(std::move(condition));
    branches.push_back(std::move(equations));
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<Expression> conditions;
  std::vector<Children<Equation>> branches;
};

//===---------------------------------------------------------------------===//
// Statements
//===---------------------------------------------------------------------===//

/// Base class of the statements.
class Statement : public Node {
public:
  using Node::Node;

  static bool classof(const ASTNode *node) {
    return node->getKind<NodeKind>() >= NodeKind::Statement &&
           node->getKind<NodeKind>() <= NodeKind::Statement_Last;
  }

  Description &getDescription() { return description; }
  const Description &getDescription() const { return description; }

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

  Description description;
};

/// Assignment `destination := value`. The destination is a component
/// reference or a `der` call on a component reference.
class AssignmentStatement
    : public NodeImpl<AssignmentStatement, Statement,
                      NodeKind::Statement_Assignment> {
public:
  using NodeImpl::NodeImpl;

  Expression *getDestination() const { return destination.get(); }
  void setDestination(std::unique_ptr<Expression> node) {
    destination = std::move(node);
  }

  Expression *getValue() const { return value.get(); }
  void setValue(std::unique_ptr<Expression> node) { value = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<Expression> destination;
  Child<Expression> value;
};

/// Assignment of multiple outputs `(a, , b) := f(x)`.
class MultipleAssignmentStatement
    : public NodeImpl<MultipleAssignmentStatement, Statement,
                      NodeKind::Statement_MultipleAssignment> {
public:
  using NodeImpl::NodeImpl;

  /// Get the destinations. Omitted destinations are null.
  const Children<Expression> &getDestinations() const { return destinations; }
  void addDestination(std::unique_ptr<Expression> node) {
    destinations.emplace_back(std::move(node));
  }

  Call *getCall() const { return call.get(); }
  void setCall(std::unique_ptr<Call> node) { call = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<Expression> destinations;
  Child<Call> call;
};

class CallStatement
    : public NodeImpl<CallStatement, Statement, NodeKind::Statement_Call> {
public:
  using NodeImpl::NodeImpl;

  Call *getCall() const { return call.get(); }
  void setCall(std::unique_ptr<Call> node) { call = std::move(node); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<Call> call;
};

class BreakStatement
    : public NodeImpl<BreakStatement, Statement, NodeKind::Statement_Break> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;
};

class ReturnStatement
    : public NodeImpl<ReturnStatement, Statement, NodeKind::Statement_Return> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;
};

/// If-statement.
class IfStatement
    : public NodeImpl<IfStatement, Statement, NodeKind::Statement_If> {
public:
  using NodeImpl::NodeImpl;

  const Children<Expression> &getConditions() const { return conditions; }
  const std::vector<Children<Statement>> &getBranches() const {
    return branches;
  }

  void addBranch(std::unique_ptr<Expression> condition,
                 Children<Statement> statements) {
    conditions.emplace_back(std::move(condition));
    branches.push_back(std::move(statements));
  }

  const Children<Statement> &getElseStatements() const {
    return elseStatements;
  }

  void setElseStatements(Children<Statement> statements) {
    elseStatements = std::move(statements);
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<Expression> conditions;
  std::vector<Children<Statement>> branches;
  Children<Statement> elseStatements;
};

/// For-statement.
class ForStatement
    : public NodeImpl<ForStatement, Statement, NodeKind::Statement_For> {
public:
  using NodeImpl::NodeImpl;

  const Children<ForIndex> &getIndices() const { return indices; }
  void addIndex(std::unique_ptr<ForIndex> node) {
    indices.emplace_back(std::move(node));
  }

  const Children<Statement> &getStatements() const { return statements; }
  void setStatements(Children<Statement> body) { statements = std::move(body); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<ForIndex> indices;
  Children<Statement> statements;
};

/// While-statement.
class WhileStatement
    : public NodeImpl<WhileStatement, Statement, NodeKind::Statement_While> {
public:
  using NodeImpl::NodeImpl;

  Expression *getCondition() const { return condition.get(); }
  void setCondition(std::unique_ptr<Expression> node) {
    condition = std::move(node);
  }

  const Children<Statement> &getStatements() const { return statements; }
  void setStatements(Children<Statement> body) { statements = std::move(body); }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Child<Expression> condition;
  Children<Statement> statements;
};

/// When-statement.
class WhenStatement
    : public NodeImpl<WhenStatement, Statement, NodeKind::Statement_When> {
public:
  using NodeImpl::NodeImpl;

  const Children<Expression> &getConditions() const { return conditions; }
  const std::vector<Children<Statement>> &getBranches() const {
    return branches;
  }

  void addBranch(std::unique_ptr<Expression> condition,
                 Children<Statement> statements) {
    conditions.emplace_back(std::move(condition));
    branches.push_back(std::move(statements));
  }

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

private:
  Children<Expression> conditions;
  std::vector<Children<Statement>> branches;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_EQUATION_H
