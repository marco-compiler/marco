#include "marco/Codegen/Transforms/ModelSolving/CyclesSymbolicSolver.h"
#include "ginac/ginac.h"
#include <string>

#include "llvm/ADT/PostOrderIterator.h"

using namespace marco::codegen;

mlir::Value cloneValueAndDependencies(mlir::OpBuilder& builder, mlir::Value value)
{
  std::stack<mlir::Operation*> cloneStack;
  std::vector<mlir::Operation*> toBeCloned;

  // Push the defining operation of the input value on the stack.
  if (auto op = value.getDefiningOp(); op != nullptr) {
    cloneStack.push(op);
  }

  // Until the stack is empty pop it, add the operand to the list, get its
  // operands (if any) and push them on the stack.
  while (!cloneStack.empty()) {
    auto op = cloneStack.top();
    cloneStack.pop();

    toBeCloned.push_back(op);

    for (const auto& operand : op->getOperands()) {
      if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
        cloneStack.push(operandOp);
      }
    }
  }

  // Clone the operations
  mlir::BlockAndValueMapping mapping;
  mlir::Operation* clonedOp = nullptr;
  for (auto opToClone : llvm::reverse(toBeCloned)) {
    clonedOp = builder.clone(*opToClone, mapping);
  }

  assert(clonedOp != nullptr);
  auto results = clonedOp->getResults();
  assert(results.size() == 1);
  return results[0];
}

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
}

// todo: is it possible to swap the ValueNode for a simple mlir::Value?
// todo: how do you make this non recursive?
GiNaC::ex visit_postorder_recursive_value(ValueNode* node, std::map<std::string, GiNaC::symbol>& nameToSymbolMap, std::map<std::string, MatchedEquation*>& nameToEquationMap, std::map<std::string, mlir::Type>& nameToTypeMap, std::map<std::string, std::vector<GiNaC::ex>>& nameToIndicesMap, MatchedEquation* matchedEquation)
{
  // If the node is the root of the tree, its value is null so it needs to be handled separately
  if (node->getFather() == nullptr) {
    GiNaC::ex lhs = visit_postorder_recursive_value(node->getChild(0), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation);
    GiNaC::ex rhs = visit_postorder_recursive_value(node->getChild(1), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation);
    return lhs == rhs;
  }

  // If the value is not a block argument, it must be a SSA value defined by an operation.
  mlir::Operation* definingOp = node->getValue().getDefiningOp();

  // If the value is defined by a constant operation, it is a leaf.
  if(auto constantOp = mlir::dyn_cast<mlir::modelica::ConstantOp>(definingOp); constantOp != nullptr) {
    mlir::Attribute attribute = constantOp.getValue();

    if (const auto integerValue = attribute.dyn_cast<mlir::IntegerAttr>()) {
      return integerValue.getInt();
    }

    if (const auto integerValue = attribute.dyn_cast<mlir::modelica::IntegerAttr>()) {
      return integerValue.getValue().getSExtValue();
    }

    GiNaC::ex res = getDoubleFromAttribute(constantOp.getValue());
    return res;
  }

  if(mlir::isa<mlir::modelica::TimeOp>(definingOp))
    return nameToSymbolMap["time"];

  if(mlir::isa<mlir::modelica::EquationSideOp>(definingOp))
    return visit_postorder_recursive_value(node->getChild(0), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation);

  if(mlir::isa<mlir::modelica::SinOp>(definingOp))
    return sin(visit_postorder_recursive_value(node->getChild(0), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation));

  if(mlir::isa<mlir::modelica::NegateOp>(definingOp))
    return -visit_postorder_recursive_value(node->getChild(0), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation);

  if(auto variableGetOp = mlir::dyn_cast<mlir::modelica::VariableGetOp>(definingOp)) {
    std::string variableName = variableGetOp.getVariable().str();

    if(!nameToSymbolMap.count(variableName)) {
      nameToSymbolMap[variableName] = GiNaC::symbol(variableName);
    }

    if(!nameToTypeMap.count(variableName)) {
      nameToTypeMap[variableName] = variableGetOp.getType();
    }

    if(!nameToIndicesMap.count(variableName)) {
      nameToIndicesMap[variableName] = {};
    }

    if (!nameToEquationMap.count(variableName)) {
      // If the this is the matched variable for the equation, add it to the array
      auto write = matchedEquation->getValueAtPath(matchedEquation->getWrite().getPath()).getDefiningOp();
      if (auto writeOp = mlir::dyn_cast<mlir::modelica::VariableGetOp>(write); writeOp == variableGetOp) {
        nameToEquationMap[variableName] = matchedEquation;
      }
    }

    return nameToSymbolMap[variableName];
  }

  if(mlir::isa<mlir::modelica::AddOp>(definingOp))
    return visit_postorder_recursive_value(node->getChild(0), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation) + visit_postorder_recursive_value(node->getChild(1), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation);

  if(mlir::isa<mlir::modelica::SubOp>(definingOp))
    return visit_postorder_recursive_value(node->getChild(0), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation) - visit_postorder_recursive_value(node->getChild(1), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation);

  if(mlir::isa<mlir::modelica::MulOp>(definingOp))
    return visit_postorder_recursive_value(node->getChild(0), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation) * visit_postorder_recursive_value(node->getChild(1), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation);

  if(mlir::isa<mlir::modelica::DivOp>(definingOp))
    return visit_postorder_recursive_value(node->getChild(0), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation) / visit_postorder_recursive_value(node->getChild(1), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation);

  if(mlir::isa<mlir::modelica::PowOp>(definingOp))
    return GiNaC::pow(
        visit_postorder_recursive_value(node->getChild(0), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation),
        visit_postorder_recursive_value(node->getChild(1), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation));

  // If we have a subscription operation, get the shape of the base vector, that should correspond with
  // the first operand. Then add an index to the symbol of the vector for each operand except the base.
  if (auto loadOp = mlir::dyn_cast<mlir::modelica::SubscriptionOp>(definingOp)) {
    if(auto subscriptionOp = mlir::dyn_cast<mlir::modelica::SubscriptionOp>(loadOp->getOperand(0).getDefiningOp())) {
      mlir::Value baseOperand = node->getChild(0)->getValue();
      std::string baseVariableName;
      if (auto variableGetOp = mlir::dyn_cast<mlir::modelica::VariableGetOp>(baseOperand.getDefiningOp())) {
        baseVariableName = variableGetOp.getVariable().str();

        std::string offset;

        std::vector<GiNaC::ex> indices;
        for (size_t i = 1; i < definingOp->getNumOperands(); ++i) {
          offset += '_';

          GiNaC::ex expression = visit_postorder_recursive_value(node->getChild(i), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, matchedEquation);
          indices.push_back(expression);

          std::ostringstream oss;
          oss << expression;
          offset += oss.str();
        }

        std::string variableName = baseVariableName + '-' + offset;

        if(!nameToSymbolMap.count(variableName)) {
          nameToSymbolMap[variableName] = GiNaC::symbol(variableName);
        }

        if(!nameToTypeMap.count(variableName)) {
          nameToTypeMap[variableName] = variableGetOp.getType();
        }

        if(!nameToIndicesMap.count(variableName)) {
          nameToIndicesMap[variableName] = indices;
        }

        if (!nameToEquationMap.count(variableName)) {
          // If the this is the matched variable for the equation, add it to the array
          auto write = matchedEquation->getValueAtPath(matchedEquation->getWrite().getPath()).getDefiningOp();
          if (auto writeOp = mlir::dyn_cast<mlir::modelica::LoadOp>(write); writeOp == loadOp) {
            nameToEquationMap[variableName] = matchedEquation;
          }
        }

        return nameToSymbolMap[variableName];
      }
      llvm_unreachable("We assume that all arrays are variables, and the arrays are fully subscripted.");
    }
    llvm_unreachable("We assume that all arrays are variables, and the arrays are fully subscripted.");
  }
  llvm_unreachable("Found operation with an unusual number of arguments\n");
}

GiNaC::ex get_equation_expression(MatchedEquation* equation, const GiNaC::lst& symbols, const GiNaC::symbol time, llvm::DenseMap<mlir::Value, GiNaC::ex> valueExpressionMap)
{
  // If the node is the root of the tree, its value is null so it needs to be handled separately

  GiNaC::ex resultExpression;

  equation->getOperation().bodyBlock()->walk(
    [&](mlir::Operation* op) {
      GiNaC::ex expression;
      mlir::Value result = op->getResult(0);

      if (mlir::modelica::EquationSidesOp sidesOp = mlir::dyn_cast<mlir::modelica::EquationSidesOp>(op)) {
        GiNaC::ex lhs = valueExpressionMap[sidesOp->getOperand(0)];
        GiNaC::ex rhs = valueExpressionMap[sidesOp->getOperand(1)];
        resultExpression = lhs == rhs;
      }

      else if(auto constantOp = mlir::dyn_cast<mlir::modelica::ConstantOp>(op); constantOp != nullptr) {
        mlir::Attribute attribute = constantOp.getValue();

        if (const auto integerValue = attribute.dyn_cast<mlir::IntegerAttr>())
          expression = integerValue.getInt();
        else if (const auto modelicaIntegerValue = attribute.dyn_cast<mlir::modelica::IntegerAttr>())
          expression = modelicaIntegerValue.getValue().getSExtValue();
        else
          expression = getDoubleFromAttribute(constantOp.getValue());
      }

      else if(mlir::isa<mlir::modelica::TimeOp>(op))
        expression = time;

      else if(mlir::isa<mlir::modelica::EquationSideOp>(op))
        expression = valueExpressionMap[op->getOperand(0)];
      else if(mlir::isa<mlir::modelica::SinOp>(op))
        expression = sin(valueExpressionMap[op->getOperand(0)]);
      else if(mlir::isa<mlir::modelica::NegateOp>(op))
        expression = -valueExpressionMap[op->getOperand(0)];

      else if(mlir::isa<mlir::modelica::AddOp>(op))
        expression = valueExpressionMap[op->getOperand(0)] + valueExpressionMap[op->getOperand(1)];
      else if(mlir::isa<mlir::modelica::SubOp>(op))
        expression = valueExpressionMap[op->getOperand(0)] - valueExpressionMap[op->getOperand(1)];
      else if(mlir::isa<mlir::modelica::MulOp>(op))
        expression = valueExpressionMap[op->getOperand(0)] * valueExpressionMap[op->getOperand(1)];
      else if(mlir::isa<mlir::modelica::DivOp>(op))
        expression = valueExpressionMap[op->getOperand(0)] / valueExpressionMap[op->getOperand(1)];

      // If we have a subscription operation, get the shape of the base vector, that should correspond with
      // the first operand. Then add an index to the symbol of the vector for each operand except the base.
      else if(mlir::isa<mlir::modelica::SubscriptionOp>(op)) {
        mlir::Value baseOperand = op->getOperand(0);
        GiNaC::ex base = valueExpressionMap[baseOperand];

        for (size_t i = 1; i < op->getNumOperands(); ++i) {
          size_t dim = baseOperand.getType().dyn_cast<mlir::modelica::ArrayType>().getShape()[i-1];
          GiNaC::idx index(valueExpressionMap[op->getOperand(i)], dim);
          base = GiNaC::indexed(base, index);
        }

        expression = base;
      }

      valueExpressionMap[result] = expression;
      return;
    }
  );

  return resultExpression;
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

// todo: check if moving the solver before the matching phase can be done
bool CyclesSymbolicSolver::solve(const std::set<MatchedEquation*>& equationSet)
{
  GiNaC::lst systemEquations;
  GiNaC::lst matchedVariables;

  std::map<std::string, GiNaC::symbol> nameToSymbolMap;
  std::map<std::string, MatchedEquation*> nameToEquationMap;
  std::map<std::string, mlir::Type> nameToTypeMap;
  std::map<std::string, std::vector<GiNaC::ex>> nameToIndicesMap;

  GiNaC::symbol time("time");
  nameToSymbolMap["time"] = time;

  GiNaC::lst trivialEquations;

  for (auto& equation : equationSet) {
    EquationValueGraph valueGraph = EquationValueGraph(equation);
    GiNaC::ex expression = visit_postorder_recursive_value(valueGraph.getEntryNode(), nameToSymbolMap, nameToEquationMap, nameToTypeMap, nameToIndicesMap, equation);
    valueGraph.erase();

    std::cerr << '\n' << "Expression: " << expression << '\n';

    // If an equation is trivial instead (e.g. x == 1), save it to later substitute it in the other ones.
    // todo: is this sufficient to cover all parameter cases or not?
    if (GiNaC::is_a<GiNaC::symbol>(expression.lhs()) && GiNaC::is_a<GiNaC::numeric>(expression.rhs())) {
      trivialEquations.append(expression);
    } else {
      systemEquations.append(expression);
    }
  }

  // The variables into wrt. which the linear system is solved are the matched ones.
  for (const auto& [name, equation] : nameToEquationMap) {
    matchedVariables.append(nameToSymbolMap[name]);
  }

  std::cerr << "System equations: " << systemEquations << '\n' << std::flush;
  std::cerr << "Matched variables: " << matchedVariables << '\n' << std::flush;

  GiNaC::ex solution = GiNaC::lsolve(systemEquations, matchedVariables);

  std::cerr << "Solution: " << solution << '\n';

  auto solutionEquations = Equations<MatchedEquation>();

  // Keep InitialEquations and Equations intact, and set the lhs as the matched variable.
  for (const auto& equation : equationSet) {
    equation->setPath(EquationPath::LEFT);
    equation->dumpIR();
//    auto simpleEquation = Equation::build(equation->getOperation(), equation->getVariables());
//    solutionEquations.add(std::make_unique<MatchedEquation>(MatchedEquation(std::move(simpleEquation), modeling::IndexSet(modeling::Point(0)), EquationPath::LEFT)));
  }

  for (const GiNaC::ex expr : solution) {
    std::string matchedVariableName;
    if (GiNaC::is_a<GiNaC::symbol>(expr.lhs())) {
      matchedVariableName = GiNaC::ex_to<GiNaC::symbol>(expr.lhs()).get_name();
    } else {
      llvm_unreachable("Expected the left hand side of the solution equation to be a symbol.");
    }

    auto equation = nameToEquationMap[matchedVariableName];
    auto equationOp = equation->getOperation();
    auto loc = equationOp.getLoc();
    builder.setInsertionPoint(equationOp);

    equationOp.bodyBlock()->erase();
    assert(equationOp.getBodyRegion().empty());
    mlir::Block* equationBodyBlock = builder.createBlock(&equationOp.getBodyRegion());
    builder.setInsertionPointToStart(equationBodyBlock);

    SymbolicVisitor visitor = SymbolicVisitor(builder, loc, equation, nameToTypeMap, nameToIndicesMap);
    expr.traverse_postorder(visitor);
  }

  return true;
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

ValueNode* ValueNode::getChild(size_t i)
{
  return this->children.at(i);
}


SymbolicVisitor::SymbolicVisitor(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    MatchedEquation* equation,
    std::map<std::string, mlir::Type>& nameToTypeMap,
    std::map<std::string, std::vector<GiNaC::ex>>& nameToIndicesMap
    ) : builder(builder), loc(loc), equation(equation), nameToTypeMap(nameToTypeMap), nameToIndicesMap(nameToIndicesMap)
{
}

void SymbolicVisitor::visit(const GiNaC::add & x) {
  mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];

  for (size_t i = 1; i < x.nops(); ++i) {
    mlir::Value rhs = expressionHashToValueMap[x.op(i).gethash()];
    mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());
    lhs = builder.create<mlir::modelica::AddOp>(loc, type, lhs, rhs);
  }

  expressionHashToValueMap[x.gethash()] = lhs;
}

void SymbolicVisitor::visit(const GiNaC::mul & x) {
  mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];

  for (size_t i = 1; i < x.nops(); ++i) {
    mlir::Value rhs = expressionHashToValueMap[x.op(i).gethash()];
    mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());
    lhs = builder.create<mlir::modelica::MulOp>(loc, type, lhs, rhs);
  }

  expressionHashToValueMap[x.gethash()] = lhs;
}

void SymbolicVisitor::visit(const GiNaC::power & x) {
  mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];
  mlir::Value rhs = expressionHashToValueMap[x.op(1).gethash()];

  mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());

  mlir::Value value = builder.create<mlir::modelica::PowOp>(loc, type, lhs, rhs);
  expressionHashToValueMap[x.gethash()] = value;
}

void SymbolicVisitor::visit(const GiNaC::function & x) {
  if (x.get_name() == "sin") {
    mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];

    mlir::Type type = lhs.getType();

    mlir::Value value = builder.create<mlir::modelica::SinOp>(loc, type, lhs);
    expressionHashToValueMap[x.gethash()] = value;
  }
}

void SymbolicVisitor::visit(const GiNaC::relational & x) {
  if (x.info(GiNaC::info_flags::relation_equal)) {
    mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];
    mlir::Value rhs = expressionHashToValueMap[x.op(1).gethash()];

    lhs = builder.create<mlir::modelica::EquationSideOp>(loc, lhs);
    rhs = builder.create<mlir::modelica::EquationSideOp>(loc, rhs);

    builder.create<mlir::modelica::EquationSidesOp>(loc, lhs, rhs);
  }
}

void SymbolicVisitor::visit(const GiNaC::numeric & x) {
  mlir::Attribute attribute;

  if (x.is_cinteger()) {
    attribute = mlir::modelica::IntegerAttr::get(builder.getContext(), x.to_int());
  } else if (x.is_real()) {
    attribute = mlir::modelica::RealAttr::get(builder.getContext(), x.to_double());
  } else {
    llvm_unreachable("Unknown variable type, aborting.");
  }

  mlir::Value value = builder.create<mlir::modelica::ConstantOp>(loc, attribute);
  expressionHashToValueMap[x.gethash()] = value;

}

void SymbolicVisitor::visit(const GiNaC::symbol & x) {
  mlir::Value value = expressionHashToValueMap[x.gethash()];
  if (value == nullptr) {
    if (x.get_name() == "time") {
      value = builder.create<mlir::modelica::TimeOp>(loc);
    } else {
      std::vector<mlir::Value> currentIndices;

      std::string variableName = x.get_name();

      for (const auto& index : nameToIndicesMap[variableName]) {
        if (expressionHashToValueMap[index.gethash()] == nullptr) {
          index.traverse_postorder(*this);
        }

        currentIndices.push_back(expressionHashToValueMap[index.gethash()]);
      }

      mlir::Type type = nameToTypeMap[x.get_name()];
      value = builder.create<mlir::modelica::VariableGetOp>(loc, type, x.get_name());

      if (!currentIndices.empty()) {
        value = builder.create<mlir::modelica::LoadOp>(loc, value, currentIndices);
      }
    }

    expressionHashToValueMap[x.gethash()] = value;
  }
}

Equations<MatchedEquation> CyclesSymbolicSolver::getSolution() const
{
  return newEquations_;
}

bool CyclesSymbolicSolver::hasUnsolvedCycles() const
{
  return !unsolvedCycles_.empty();
}

Equations<MatchedEquation> CyclesSymbolicSolver::getUnsolvedEquations() const
{
  Equations<MatchedEquation> result;

  for (const auto& equation : unsolvedCycles_) {
    modeling::IndexSet indices(equation->getIterationRanges());

    for (const auto& range : llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
      result.add(std::make_unique<MatchedEquation>(
          equation->clone(), modeling::IndexSet(range), equation->getWrite().getPath()));
    }
  }

  return result;
}
