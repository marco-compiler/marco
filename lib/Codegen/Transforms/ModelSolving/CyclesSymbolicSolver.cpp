#include "marco/Codegen/Transforms/ModelSolving/CyclesSymbolicSolver.h"
#include "ginac/ginac.h"
#include <string>
//#include <symengine/expression.h>

#include "llvm/ADT/PostOrderIterator.h"

using namespace marco::codegen;

mlir::Operation* OperationNode::getOperation()
{
  return operation;
}

OperationNode::OperationNode(
     mlir::Operation* operatwalion,
     OperationNode* next,
     OperationNode* prev,
     OperationNode* father,
     OperationNode* child,
     size_t childNumber,
     size_t numberOfChildren) :
     operation(operation),
     next(next),
     prev(prev),
     father(father),
     child(child),
     childNumber(childNumber),
     numberOfChildren(numberOfChildren)
{
}

void OperationNode::setNext(OperationNode* next)
{
  this->next = next;
}
void OperationNode::setChild(OperationNode* child)
{
  this->child = child;
}

OperationNode* OperationNode::getChild()
{
  return this->child;
}

OperationNode* OperationNode::getNext()
{
  return this->next;
}

//EquationSideGraph::EquationSideGraph(MatchedEquation* equation) : equation(equation)
//{
//  std::stack<OperationNode*> stack;
//
//  // The terminator always has two children, the LHS and the RHS
//  entryNode = new OperationNode(equation->getOperation().bodyBlock()->getTerminator(),
//                                nullptr, nullptr, nullptr, nullptr, 0, 2);
//
//  stack.push(entryNode);
//
//  while (!stack.empty()) {
//    auto father = stack.top();
//    stack.pop();
//
//    OperationNode* prev = nullptr;
//    size_t childNumber = 0;
//    for (const auto& operand : father->getOperation()->getOperands()) {
//      if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
//        size_t numberOfChildren = operandOp->getOperands().size();
//
//        auto newNode = new OperationNode(operandOp, nullptr, prev, father, nullptr,
//                                         childNumber,  numberOfChildren);
//
//        if (prev)
//          prev->setNext(newNode);
//        prev = newNode;
//
//        if(childNumber == 0)
//          father->setChild(newNode);
//        ++childNumber;
//
//        stack.push(newNode);
//      }
//    }
//  }
//}

EquationValueGraph::EquationValueGraph(MatchedEquation* equation) : equation(equation)
{
  std::stack<ValueNode*> stack;

  mlir::Operation* terminator = equation->getOperation().bodyBlock()->getTerminator();

  mlir::Value lhs = terminator->getOperand(0);
  mlir::Value rhs = terminator->getOperand(1);

  auto* lhsNode = new ValueNode(lhs, &entryNode);
  auto* rhsNode = new ValueNode(rhs, &entryNode);

  stack.push(lhsNode);
  stack.push(rhsNode);

  entryNode.addChild(lhsNode);
  entryNode.addChild(rhsNode);

  while (!stack.empty()) {
    auto father = stack.top();
    stack.pop();

    // If the value is defined by an operation, take its operand values and add them as children.
    // If instead the value is a block argument, it will have no children.
    if (mlir::Operation* operandOp = father->getValue().getDefiningOp(); operandOp != nullptr) {

      for (const mlir::Value operand : operandOp->getOperands()) {
        auto* newNode = new ValueNode(operand, father);

        father->addChild(newNode);
        stack.push(newNode);
      }
    }
  }
}

ValueNode* EquationValueGraph::getEntryNode()
{
  return &this->entryNode;
}

static void deleteValueNode(ValueNode* node) {
  if (node->getFather() != nullptr) {
    delete node;
  }
}

void EquationValueGraph::erase()
{
  walk(deleteValueNode);
}

static void printValueNode(ValueNode* node) {
  if (node->getFather() != nullptr) {
    if (mlir::Operation* operandOp = node->getValue().getDefiningOp()) {
      operandOp->dump();
    }
  }
}

void EquationValueGraph::print()
{
  walk(printValueNode);
}

void EquationValueGraph::walk(void (*func)(ValueNode*))
{
  std::stack<ValueNode*> stack;
  std::vector<ValueNode*> vector;

  stack.push(&entryNode);

  while (!stack.empty()) {
    auto father = stack.top();
    stack.pop();

    vector.push_back(father);

    for (ValueNode* child : father->getChildren()) {
      stack.push(child);
    }
  }

  for (ValueNode* node : llvm::reverse(vector)) {
    func(node);
  }
}

EquationGraph::EquationGraph(MatchedEquation* equation) : equation(equation)
{
  std::stack<OperationNode*> stack;

  // The terminator always has two children, the LHS and the RHS
  entryNode = new OperationNode(equation->getOperation().bodyBlock()->getTerminator(),
                                nullptr, nullptr, nullptr, nullptr, 0, 2);

  stack.push(entryNode);

  while (!stack.empty()) {
    auto father = stack.top();
    stack.pop();

    OperationNode* prev = nullptr;
    size_t childNumber = 0;
    for (const auto& operand : father->getOperation()->getOperands()) {
      if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
        size_t numberOfChildren = operandOp->getOperands().size();

        auto newNode = new OperationNode(operandOp, nullptr, prev, father, nullptr,
                                         childNumber,  numberOfChildren);

        if (prev)
          prev->setNext(newNode);
        prev = newNode;

        if(childNumber == 0)
          father->setChild(newNode);
        ++childNumber;

        stack.push(newNode);
      }
    }
  }
}

OperationNode* EquationGraph::getEntryNode()
{
  return entryNode;
}

void EquationGraph::print()
{
  std::stack<OperationNode*> stack;

  stack.push(entryNode);

  while (!stack.empty()) {
    auto father = stack.top();
    stack.pop();

    auto child = father->getChild();
    while (child) {
      stack.push(child);
      child = child->getNext();
    }

    father->getOperation()->dump();
  }
}

void print_postorder(OperationNode* node)
{
  if (node != nullptr){
    auto left = node->getChild();
    print_postorder(left);
    if (left != nullptr) {
      auto right = left->getNext();
      print_postorder(right);
    }
    node->getOperation()->dump();
  }
}

static double getDoubleFromAttribute(mlir::Attribute attribute)
{
  if (auto indexAttr = attribute.dyn_cast<mlir::IntegerAttr>()) {
    return indexAttr.getInt();
  }

  if (auto booleanAttr = attribute.dyn_cast<mlir::modelica::BooleanAttr>()) {
    return booleanAttr.getValue() ? 1 : 0;
  }

  if (auto integerAttr = attribute.dyn_cast<mlir::modelica::IntegerAttr>()) {
    return integerAttr.getValue().getSExtValue();
  }

  if (auto realAttr = attribute.dyn_cast<mlir::modelica::RealAttr>()) {
    return realAttr.getValue().convertToDouble();
  }

  llvm_unreachable("Unknown attribute type");
  return 0;
}

GiNaC::ex build_binary_operation(const GiNaC::ex& op1,const GiNaC::ex& op2, mlir::Operation* op) {
  if(mlir::isa<mlir::modelica::AddOp>(op))
    return op1 + op2;
  if(mlir::isa<mlir::modelica::SubOp>(op))
    return op1 - op2;
  if(mlir::isa<mlir::modelica::EquationSidesOp>(op))
    return op1 == op2;
  if(mlir::isa<mlir::modelica::SubscriptionOp>(op))
    //todo
    return op1 == op2;

  op->dump();
  llvm_unreachable("Unrecognized binary operation: ");
}

GiNaC::ex build_unary_operation(const GiNaC::ex& op1, mlir::Operation* op) {
  if(mlir::isa<mlir::modelica::EquationSideOp>(op)) {
    return op1;
  }

  if(mlir::isa<mlir::modelica::LoadOp>(op)) {
    return op1;
  }

  op->dump();
  llvm_unreachable("Unrecognized unary operation.");

}

GiNaC::ex visit_postorder_recursive(OperationNode* node, const GiNaC::lst& symbols)
{
  std::cerr << "Visiting postorder\n" << std::flush;
  if (node != nullptr) {
    node->getOperation()->dump();

    if (auto loadOp = mlir::dyn_cast<mlir::modelica::LoadOp>(node->getOperation())) {
      auto arg = mlir::dyn_cast<mlir::BlockArgument>(node->getOperation()->getOperand(0));
      if (arg) {
        std::cerr << "Found block argument number " + std::to_string(arg.getArgNumber()) + '\n';
        return symbols[arg.getArgNumber()];
      }
    } else if (auto subscriptionOp = mlir::dyn_cast<mlir::modelica::SubscriptionOp>(node->getOperation())) {
      auto arg = mlir::dyn_cast<mlir::BlockArgument>(node->getOperation()->getOperand(0));
      if (arg) {
        GiNaC::ex sym = symbols[arg.getArgNumber()];
        auto operands = node->getOperation()->getOperands();
        auto max_rank = operands.size();

        for (size_t rank = 0; rank < max_rank; ++rank) {
          //sym = GiNaC::indexed(sym, GiNaC::idx(GiNaC::symbol("idx" + std::to_string(rank)), ));
        }
        std::cerr << "Found block argument number " + std::to_string(arg.getArgNumber()) + '\n';
        return symbols[arg.getArgNumber()];
      }
    }

    OperationNode* left = node->getChild();
    GiNaC::ex left_ex;
    GiNaC::ex right_ex;

    OperationNode* right = nullptr;
    if (left != nullptr) {
      left_ex = visit_postorder_recursive(left, symbols);

      std::cerr << "left expression: " << left_ex << '\n' << std::flush;
      right = left->getNext();
    }

    // Build the operation expression with the Symbolic solver
    auto op = node->getOperation();

    if (left != nullptr) {
      if (right != nullptr) {
        right_ex = visit_postorder_recursive(right, symbols);
        std::cerr << "right expression: " << right_ex << '\n' << std::flush;
        return build_binary_operation(left_ex, right_ex, op);
      }
      return build_unary_operation(left_ex, op);
    }
    if(mlir::isa<mlir::modelica::ConstantOp>(op)) {
      std::cerr << "ConstantOp\n" << std::flush;
      auto constantOp = mlir::dyn_cast<mlir::modelica::ConstantOp>(op);
      GiNaC::ex res = getDoubleFromAttribute(constantOp.getValue());
      return res;
    }
    op->dump();
    std::cerr << "NOT GOOD\n";
    llvm_unreachable("Unreachable control flow");
  }

  std::cerr << "Null node\n" << std::flush;
}


void EquationGraph::erase()
{
  std::stack<OperationNode*> stack;
  std::vector<OperationNode*> vector;

  stack.push(entryNode);

  while (!stack.empty()) {
    auto father = stack.top();
    stack.pop();

    vector.push_back(father);

    auto child = father->getChild();
    while (child) {
      stack.push(child);
      child = child->getNext();
    }
  }

  for (const auto node : llvm::reverse(vector)) {
    delete node;
  }
}

CyclesSymbolicSolver::CyclesSymbolicSolver(mlir::OpBuilder& builder) : builder(builder)
{

}

void visit_postorder(const MatchedEquation& equation, const GiNaC::lst& symbols) {
  std::stack<mlir::Value> valueStack;
  std::stack<mlir::Operation*> operationStack;

  std::stack<mlir::Operation*> tempStack;

  mlir::Operation* terminator = equation.getOperation().bodyBlock()->getTerminator();
  mlir::ValueRange operands = terminator->getOperands();

  tempStack.push(terminator);
  operationStack.push(terminator);

  while (!tempStack.empty()) {
    mlir::Operation* operation = tempStack.top();
    tempStack.pop();

    for (const mlir::Value& operand : operation->getOperands()) {
      valueStack.push(operand);
      tempStack.push(operand.getDefiningOp());
      operationStack.push(operand.getDefiningOp());
    }
  }

}

bool CyclesSymbolicSolver::solve(Model<MatchedEquation>& model)
{
  model.getOperation()->dump();
  // The list of equations among which the cycles have to be searched
  llvm::SmallVector<MatchedEquation*> toBeProcessed;

  // The first iteration will use all the equations of the model
  for (const auto& equation : model.getEquations()) {
    toBeProcessed.push_back(equation.get());
  }

  auto systemEquations = GiNaC::lst();

  size_t numberOfScalarEquations = 0;

  for (const auto& equation : toBeProcessed) {
    numberOfScalarEquations += equation->getIterationRanges().flatSize();
  }

  size_t numberOfArguments = model.getOperation().getBodyRegion().getArguments().size();

  auto symbols = GiNaC::lst();
  for (size_t i = 0; i < numberOfArguments; ++i) {
    GiNaC::symbol sym("sym" + std::to_string(i));
    symbols.append(sym);
  }

  for (const auto& equation : toBeProcessed) {
    std::cerr << "Num operands: " << numberOfScalarEquations << std::flush;

    equation->getOperation()->dump();

    EquationValueGraph valueGraph = EquationValueGraph(equation);

    std::cerr << "Equation Value Graph built\n" << std::flush;

    valueGraph.print();

    exit(1);

    auto graph = EquationGraph(equation);

    GiNaC::ex expression = visit_postorder_recursive(graph.getEntryNode(), symbols);

    graph.erase();
    systemEquations.append(expression);
    std::cerr << "This is the complete expression: " << expression << '\n' << std::flush;
  }

  GiNaC::lst variables = {};

  for (const auto& symbol : symbols) {
    variables.append(symbol);
  }

  std::cerr << systemEquations;
  std::cerr << "\nSOLUTION:\n" << std::flush;
  std::cerr << GiNaC::lsolve(systemEquations, variables) << "\n" << std::flush;

  return false;
}

ValueNode::ValueNode(mlir::Value value, ValueNode* father)
{
  this->value = value;
  this->father = father;
}

mlir::Value ValueNode::getValue()
{
  return this->value;
}

std::vector<ValueNode*>& ValueNode::getChildren()
{
  return this->children;
}

void ValueNode::addChild(ValueNode* child)
{
  this->children.push_back(child);
}

ValueNode* ValueNode::getFather()
{
  return this->father;
}
