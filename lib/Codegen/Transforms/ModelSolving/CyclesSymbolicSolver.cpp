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

OperationNode::OperationNode(mlir::Operation* operation,
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

  op->dump();
  llvm_unreachable("Unrecognized binary operation: ");
}

GiNaC::ex build_unary_operation(const GiNaC::ex& op1, mlir::Operation* op) {
  if(mlir::isa<mlir::modelica::EquationSideOp>(op)) {
    std::cout << "SidesOp\n" << std::flush;
    return op1;
  }

  llvm_unreachable("Unrecognized unary operation.");

}

GiNaC::ex visit_postorder(OperationNode* node, const std::vector<GiNaC::symbol>& symbols)
{
  std::cout << "Visiting postorder\n" << std::flush;
  if (node != nullptr) {
    node->getOperation()->dump();
    if (!node->getOperation()->getOperands().empty()) {
      auto arg = mlir::dyn_cast<mlir::BlockArgument>(node->getOperation()->getOperand(0));
      if (arg) {
        std::cerr << "Found block argument number " + std::to_string(arg.getArgNumber()) + '\n';
        return symbols[arg.getArgNumber()];
      }
    }

    OperationNode* left = node->getChild();
    GiNaC::ex left_ex;
    GiNaC::ex right_ex;
    OperationNode* right = nullptr;
    if (left != nullptr) {
      left_ex = visit_postorder(left, symbols);
      std::cout << "left expression: " << left_ex << '\n' << std::flush;
      right = left->getNext();
    }

    // Build the operation expression with the Symbolic solver
    auto op = node->getOperation();

    if (left != nullptr) {
      if (right != nullptr) {
        right_ex = visit_postorder(right, symbols);
        std::cout << "right expression: " << right_ex << '\n' << std::flush;
        return build_binary_operation(left_ex, right_ex, op);
      }
      return build_unary_operation(left_ex, op);
    }
    if(mlir::isa<mlir::modelica::ConstantOp>(op)) {
      std::cout << "ConstantOp\n" << std::flush;
      auto constantOp = mlir::dyn_cast<mlir::modelica::ConstantOp>(op);
      GiNaC::ex res = getDoubleFromAttribute(constantOp.getValue());
      return res;
    }
    op->dump();
    llvm_unreachable("Unreachable control flow");
  }

  std::cout << "Null node\n" << std::flush;
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

void postorder_traversal(EquationGraph& equationGraph) {

}

bool CyclesSymbolicSolver::solve(Model<MatchedEquation>& model)
{
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

  auto symbols = std::vector<GiNaC::symbol>();
  for (size_t i = 0; i < numberOfScalarEquations; ++i) {
    GiNaC::symbol sym("s" + std::to_string(i));
    symbols.push_back(sym);
  }

  for (const auto& equation : toBeProcessed) {
    std::cout << "Num operands: " << numberOfScalarEquations << std::flush;

    auto graph = EquationGraph(equation);
    //print_postorder(graph.getEntryNode());
    GiNaC::ex expression = visit_postorder(graph.getEntryNode(), symbols);
    graph.erase();
    systemEquations.append(expression);
    std::cout << "This is the complete expression: " << expression << '\n' << std::flush;
  }

  GiNaC::lst variables = {};

  for (const auto& symbol : symbols) {
    variables.append(symbol);
  }

  std::cout << systemEquations;
  std::cout << "\nSOLUTION:\n" << std::flush;
  std::cout << GiNaC::lsolve(systemEquations, variables) << "\n" << std::flush;

  return false;
}


