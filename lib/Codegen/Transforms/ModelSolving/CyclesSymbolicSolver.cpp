#include "marco/Codegen/Transforms/ModelSolving/CyclesSymbolicSolver.h"

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

CyclesSymbolicSolver::CyclesSymbolicSolver(mlir::OpBuilder& builder) : builder(builder)
{

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

  return false;
}


