#include "marco/Codegen/Transforms/ModelSolving/CyclesSymbolicSolver.h"
#include "ginac/ginac.h"
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

    father->getOperation()->dump();

    auto child = father->getChild();
    while (child) {
      stack.push(child);
      child = child->getNext();
    }
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



GiNaC::ex visit_postorder(OperationNode* node)
{
  if (node != nullptr){
    auto left = node->getChild();
    GiNaC::ex left_ex = visit_postorder(left);
    GiNaC::ex right_ex;
    if (left != nullptr) {
      auto right = left->getNext();
      right_ex = visit_postorder(right);
    }
    auto op = node->getOperation();
    if (left != nullptr) {
      
    }

  }
}

GiNaC::ex build_binary_operation() {
  
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

  auto eq = toBeProcessed[0];
  eq->dumpIR();
  auto graph = EquationGraph(toBeProcessed[0]);

  graph.print();

  print_postorder(graph.getEntryNode());

  graph.erase();

  return false;
}


