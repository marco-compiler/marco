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

// todo: how do you make this non recursive?
GiNaC::ex visit_postorder_recursive_value(mlir::Value value, MatchedEquation* matchedEquation, std::map<std::string, SymbolInfo>& symbolNameToInfoMap)
{
  // If the value is not a block argument, it must be a SSA value defined by an operation.
  mlir::Operation* definingOp = value.getDefiningOp();

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
    return symbolNameToInfoMap["time"].symbol;

  if(mlir::isa<mlir::modelica::EquationSideOp>(definingOp))
    return visit_postorder_recursive_value(definingOp->getOperand(0), matchedEquation, symbolNameToInfoMap);

  if(mlir::isa<mlir::modelica::SinOp>(definingOp))
    return sin(visit_postorder_recursive_value(definingOp->getOperand(0), matchedEquation, symbolNameToInfoMap));

  if(mlir::isa<mlir::modelica::NegateOp>(definingOp))
    return -visit_postorder_recursive_value(definingOp->getOperand(0), matchedEquation, symbolNameToInfoMap);

  if(auto variableGetOp = mlir::dyn_cast<mlir::modelica::VariableGetOp>(definingOp)) {
    std::string variableName = variableGetOp.getVariable().str();

    if(!symbolNameToInfoMap.count(variableName)) {
      symbolNameToInfoMap[variableName] = SymbolInfo();
      symbolNameToInfoMap[variableName].symbol = GiNaC::symbol(variableName);
      symbolNameToInfoMap[variableName].variableName = variableName;
      symbolNameToInfoMap[variableName].variableType = variableGetOp.getType();
      symbolNameToInfoMap[variableName].indices = {};
    }

    if (symbolNameToInfoMap[variableName].matchedEquation == nullptr) {
      // If the this is the matched variable for the equation, add it to the array
      auto write = matchedEquation->getValueAtPath(matchedEquation->getWrite().getPath()).getDefiningOp();
      if (auto writeOp = mlir::dyn_cast<mlir::modelica::VariableGetOp>(write); writeOp == variableGetOp) {
        symbolNameToInfoMap[variableName].matchedEquation = matchedEquation;
      }
    }

    return symbolNameToInfoMap[variableName].symbol;
  }

  if(mlir::isa<mlir::modelica::AddOp>(definingOp))
    return visit_postorder_recursive_value(definingOp->getOperand(0), matchedEquation, symbolNameToInfoMap) +
        visit_postorder_recursive_value(definingOp->getOperand(1), matchedEquation, symbolNameToInfoMap);

  if(mlir::isa<mlir::modelica::SubOp>(definingOp))
    return visit_postorder_recursive_value(definingOp->getOperand(0), matchedEquation, symbolNameToInfoMap) -
        visit_postorder_recursive_value(definingOp->getOperand(1), matchedEquation, symbolNameToInfoMap);

  if(mlir::isa<mlir::modelica::MulOp>(definingOp))
    return visit_postorder_recursive_value(definingOp->getOperand(0), matchedEquation, symbolNameToInfoMap) *
        visit_postorder_recursive_value(definingOp->getOperand(1), matchedEquation, symbolNameToInfoMap);

  if(mlir::isa<mlir::modelica::DivOp>(definingOp))
    return visit_postorder_recursive_value(definingOp->getOperand(0), matchedEquation, symbolNameToInfoMap) /
        visit_postorder_recursive_value(definingOp->getOperand(1), matchedEquation, symbolNameToInfoMap);

  if(mlir::isa<mlir::modelica::PowOp>(definingOp))
    return GiNaC::pow(
        visit_postorder_recursive_value(definingOp->getOperand(0), matchedEquation, symbolNameToInfoMap),
        visit_postorder_recursive_value(definingOp->getOperand(1), matchedEquation, symbolNameToInfoMap));

  // If we have a subscription operation, get the shape of the base vector, that should correspond with
  // the first operand. Then add an index to the symbol of the vector for each operand except the base.
  if (auto loadOp = mlir::dyn_cast<mlir::modelica::LoadOp>(definingOp)) {
    if(auto subscriptionOp = mlir::dyn_cast<mlir::modelica::SubscriptionOp>(loadOp->getOperand(0).getDefiningOp())) {
      mlir::Value baseOperand = subscriptionOp->getOperand(0);
      std::string baseVariableName;
      matchedEquation->dumpIR();
      if (auto variableGetOp = mlir::dyn_cast<mlir::modelica::VariableGetOp>(baseOperand.getDefiningOp())) {
        baseVariableName = variableGetOp.getVariable().str();

        std::string offset;

        std::vector<GiNaC::ex> indices;
        for (size_t i = 1; i < subscriptionOp->getNumOperands(); ++i) {
          if (offset.length())
            offset += '_';

          GiNaC::ex expression = visit_postorder_recursive_value(definingOp->getOperand(0).getDefiningOp()->getOperand(i), matchedEquation, symbolNameToInfoMap);
          indices.push_back(expression);

          std::ostringstream oss;
          oss << expression;
          offset += oss.str();
        }

        std::string variableName = baseVariableName + '_' + offset;

        if(!symbolNameToInfoMap.count(variableName)) {
          symbolNameToInfoMap[variableName] = SymbolInfo();
          symbolNameToInfoMap[variableName].symbol = GiNaC::symbol(variableName);
          symbolNameToInfoMap[variableName].variableName = baseVariableName;
          symbolNameToInfoMap[variableName].variableType = variableGetOp.getType();
          symbolNameToInfoMap[variableName].indices = indices;
        }

        if (symbolNameToInfoMap[variableName].matchedEquation == nullptr) {
          // If the this is the matched variable for the equation, add it to the array
          auto write = matchedEquation->getValueAtPath(matchedEquation->getWrite().getPath()).getDefiningOp();
          std::cerr << "WRITE OP:\n";
          write->dump();
          std::cerr << "VARIABLE GET OP:\n";
          variableGetOp->dump();
          if (auto writeOp = mlir::dyn_cast<mlir::modelica::LoadOp>(write); writeOp == loadOp) {
            std::cerr << "MATCHED OP:\n";
            write->dump();
            symbolNameToInfoMap[variableName].matchedEquation = matchedEquation;
          }
        }

        return symbolNameToInfoMap[variableName].symbol;
      }
      matchedEquation->dumpIR();
      baseOperand.getDefiningOp()->dump();
      llvm_unreachable("Not a VariableGetOp.");
    }
    matchedEquation->dumpIR();
    definingOp->dump();
    llvm_unreachable("Not a SubscriptionOp.");
  }
  matchedEquation->dumpIR();
  definingOp->dump();
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

bool CyclesSymbolicSolver::solve(const std::set<MatchedEquation*>& equationSet)
{
  GiNaC::lst systemEquations;
  GiNaC::lst matchedVariables;

  std::map<std::string, SymbolInfo> symbolNameToInfoMap;

  symbolNameToInfoMap["time"] = SymbolInfo();
  symbolNameToInfoMap["time"].symbol = GiNaC::symbol("time");
  symbolNameToInfoMap["time"].variableName = "time";
  symbolNameToInfoMap["time"].indices = {};
  symbolNameToInfoMap["time"].matchedEquation = nullptr;

  GiNaC::lst trivialEquations;

  for (auto& equation : equationSet) {
    auto terminator = mlir::cast<mlir::modelica::EquationSidesOp>(equation->getOperation().bodyBlock()->getTerminator());

    GiNaC::ex lhs = visit_postorder_recursive_value(terminator.getLhsValues()[0], equation, symbolNameToInfoMap);
    GiNaC::ex rhs = visit_postorder_recursive_value(terminator.getRhsValues()[0], equation, symbolNameToInfoMap);

    GiNaC::ex expression = lhs == rhs;

    std::cerr << '\n' << "Expression: " << expression << '\n';

//    // If an equation is trivial instead (e.g. x == 1), save it to later substitute it in the other ones.
    if (GiNaC::is_a<GiNaC::symbol>(expression.lhs()) && GiNaC::is_a<GiNaC::numeric>(expression.rhs())) {
      trivialEquations.append(expression);
    } else {
      systemEquations.append(expression);
    }
  }

  // The variables into wrt. which the linear system is solved are the matched ones.
  for (const auto& [name, info] : symbolNameToInfoMap) {
    if (info.matchedEquation != nullptr) {
      matchedVariables.append(info.symbol);
    }
  }

  std::cerr << "System equations: " << systemEquations << '\n' << std::flush;
  std::cerr << "Matched variables: " << matchedVariables << '\n' << std::flush;

  GiNaC::ex solution;
  try {
    solution = GiNaC::lsolve(systemEquations, matchedVariables);
  } catch (std::logic_error& e) {
      // The system is not linear so it cannot be solved by the symbolic solver.
    std::cerr << "The system of equations is not linear" << std::endl;
    return false;
  };

  std::cerr << "Solution: " << solution << '\n';

  if (solution.gethash() != systemEquations.gethash()) {
    for (const GiNaC::ex expr : solution) {
      std::string matchedVariableName;
      if (GiNaC::is_a<GiNaC::symbol>(expr.lhs())) {
        matchedVariableName = GiNaC::ex_to<GiNaC::symbol>(expr.lhs()).get_name();
      } else {
        llvm_unreachable("Expected the left hand side of the solution equation to be a symbol.");
      }

      auto equation = symbolNameToInfoMap[matchedVariableName].matchedEquation;

      equation->setPath(EquationPath::LEFT);
      // todo: how does one modify the EquationOp attributes?
//      equation->getOperation()->setAttr("path", );

      auto equationOp = equation->getOperation();
      auto loc = equationOp.getLoc();
      builder.setInsertionPoint(equationOp);

      equationOp.bodyBlock()->erase();
      assert(equationOp.getBodyRegion().empty());
      mlir::Block* equationBodyBlock = builder.createBlock(&equationOp.getBodyRegion());
      builder.setInsertionPointToStart(equationBodyBlock);

      SymbolicVisitor visitor = SymbolicVisitor(builder, loc, equation, symbolNameToInfoMap);
      expr.traverse_postorder(visitor);

      equation->dumpIR();
      equation->getWrite().getPath().dump();
      auto simpleEquation = Equation::build(equation->getOperation(), equation->getVariables());
      newEquations_.add(std::make_unique<MatchedEquation>(MatchedEquation(std::move(simpleEquation), equation->getIterationRanges(), equation->getWrite().getPath())));
      solvedEquations_.push_back(equation);
    }

    return true;
  }

  std::cerr << "The system of equations was already solved." << std::endl;
  return true;
}

SymbolicVisitor::SymbolicVisitor(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    MatchedEquation* equation,
    std::map<std::string, SymbolInfo>& symbolNameToInfoMap
    ) : builder(builder), loc(loc), equation(equation), symbolNameToInfoMap(symbolNameToInfoMap)
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

      for (const auto& index : symbolNameToInfoMap[variableName].indices) {
        if (expressionHashToValueMap[index.gethash()] == nullptr) {
          index.traverse_postorder(*this);
        }

        currentIndices.push_back(expressionHashToValueMap[index.gethash()]);
      }

      mlir::Type type = symbolNameToInfoMap[x.get_name()].variableType;
      std::string baseVariableName = symbolNameToInfoMap[variableName].variableName;
      value = builder.create<mlir::modelica::VariableGetOp>(loc, type, baseVariableName);

      if (!currentIndices.empty()) {
        value = builder.create<mlir::modelica::SubscriptionOp>(loc, value, currentIndices);
        value = builder.create<mlir::modelica::LoadOp>(loc, value);
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