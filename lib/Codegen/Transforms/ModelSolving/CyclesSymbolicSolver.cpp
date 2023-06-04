#include "marco/Codegen/Transforms/ModelSolving/CyclesSymbolicSolver.h"
#include "ginac/ginac.h"
#include <string>

#include "llvm/ADT/PostOrderIterator.h"

using namespace marco::codegen;

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

GiNaC::ex getExpressionFromEquation(MatchedEquation* matchedEquation, std::map<std::string, SymbolInfo>& symbolNameToInfoMap) {
  auto terminator = mlir::dyn_cast<mlir::modelica::EquationSidesOp>(matchedEquation->getOperation().bodyBlock()->getTerminator());

  GiNaC::ex solution;
  ModelicaToSymbolicEquationVisitor visitor(matchedEquation, symbolNameToInfoMap, solution);

  auto operation = matchedEquation->getOperation().bodyBlock()->begin();
  while (operation != matchedEquation->getOperation().bodyBlock()->end()) {
    if(auto variableGetOp = mlir::dyn_cast<mlir::modelica::VariableGetOp>(operation)) {
      visitor.visit(variableGetOp);
    } else if(auto subscriptionOp = mlir::dyn_cast<mlir::modelica::SubscriptionOp>(operation)) {
      visitor.visit(subscriptionOp);
    } else if(auto loadOp = mlir::dyn_cast<mlir::modelica::LoadOp>(operation)) {
      visitor.visit(loadOp);
    } else if(auto constantOp = mlir::dyn_cast<mlir::modelica::ConstantOp>(operation)) {
      visitor.visit(constantOp);
    } else if(auto timeOp = mlir::dyn_cast<mlir::modelica::TimeOp>(operation)) {
      visitor.visit(timeOp);
    } else if(auto negateOp = mlir::dyn_cast<mlir::modelica::NegateOp>(operation)) {
      visitor.visit(negateOp);
    } else if(auto addOp = mlir::dyn_cast<mlir::modelica::AddOp>(operation)) {
      visitor.visit(addOp);
    } else if(auto subOp = mlir::dyn_cast<mlir::modelica::SubOp>(operation)) {
      visitor.visit(subOp);
    } else if(auto mulOp = mlir::dyn_cast<mlir::modelica::MulOp>(operation)) {
      visitor.visit(mulOp);
    } else if(auto divOp = mlir::dyn_cast<mlir::modelica::DivOp>(operation)) {
      visitor.visit(divOp);
    } else if(auto powOp = mlir::dyn_cast<mlir::modelica::PowOp>(operation)) {
      visitor.visit(powOp);
    } else if(auto sinOp = mlir::dyn_cast<mlir::modelica::SinOp>(operation)) {
      visitor.visit(sinOp);
    } else if(auto equationSideOp = mlir::dyn_cast<mlir::modelica::EquationSideOp>(operation)) {
      visitor.visit(equationSideOp);
    } else if(auto equationSidesOp = mlir::dyn_cast<mlir::modelica::EquationSidesOp>(operation)) {
      visitor.visit(equationSidesOp);
    }

    ++operation;
  }

  return solution;
}

CyclesSymbolicSolver::CyclesSymbolicSolver(mlir::OpBuilder& builder) : builder(builder)
{
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

    GiNaC::ex expression = getExpressionFromEquation(equation, symbolNameToInfoMap);

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

      SymbolicToModelicaEquationVisitor visitor = SymbolicToModelicaEquationVisitor(builder, loc, equation, symbolNameToInfoMap);
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

SymbolicToModelicaEquationVisitor::SymbolicToModelicaEquationVisitor(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    MatchedEquation* equation,
    std::map<std::string, SymbolInfo>& symbolNameToInfoMap
    ) : builder(builder), loc(loc), equation(equation), symbolNameToInfoMap(symbolNameToInfoMap)
{
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::add & x) {
  mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];

  for (size_t i = 1; i < x.nops(); ++i) {
    mlir::Value rhs = expressionHashToValueMap[x.op(i).gethash()];
    mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());
    lhs = builder.create<mlir::modelica::AddOp>(loc, type, lhs, rhs);
  }

  expressionHashToValueMap[x.gethash()] = lhs;
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::mul & x) {
  mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];

  for (size_t i = 1; i < x.nops(); ++i) {
    mlir::Value rhs = expressionHashToValueMap[x.op(i).gethash()];
    mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());
    lhs = builder.create<mlir::modelica::MulOp>(loc, type, lhs, rhs);
  }

  expressionHashToValueMap[x.gethash()] = lhs;
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::power & x) {
  mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];
  mlir::Value rhs = expressionHashToValueMap[x.op(1).gethash()];

  mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());

  mlir::Value value = builder.create<mlir::modelica::PowOp>(loc, type, lhs, rhs);
  expressionHashToValueMap[x.gethash()] = value;
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::function & x) {
  if (x.get_name() == "sin") {
    mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];

    mlir::Type type = lhs.getType();

    mlir::Value value = builder.create<mlir::modelica::SinOp>(loc, type, lhs);
    expressionHashToValueMap[x.gethash()] = value;
  }
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::relational & x) {
  if (x.info(GiNaC::info_flags::relation_equal)) {
    mlir::Value lhs = expressionHashToValueMap[x.op(0).gethash()];
    mlir::Value rhs = expressionHashToValueMap[x.op(1).gethash()];

    lhs = builder.create<mlir::modelica::EquationSideOp>(loc, lhs);
    rhs = builder.create<mlir::modelica::EquationSideOp>(loc, rhs);

    builder.create<mlir::modelica::EquationSidesOp>(loc, lhs, rhs);
  }
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::numeric & x) {
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

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::symbol & x) {
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

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::VariableGetOp variableGetOp)
{
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

  valueToExpressionMap[variableGetOp.getResult()] = symbolNameToInfoMap[variableName].symbol;
}

void ModelicaToSymbolicEquationVisitor::visit(const mlir::modelica::SubscriptionOp subscriptionOp)
{
  // As of now we don't need to do anything when we visit a SubscriptionOp, as array loading is managed by LoadOp
}

void ModelicaToSymbolicEquationVisitor::visit(const mlir::modelica::LoadOp loadOp)
{
  if(auto subscriptionOp = mlir::dyn_cast<mlir::modelica::SubscriptionOp>(loadOp->getOperand(0).getDefiningOp())) {
    if (auto variableGetOp = mlir::dyn_cast<mlir::modelica::VariableGetOp>(subscriptionOp->getOperand(0).getDefiningOp())) {
      std::string baseVariableName = variableGetOp.getVariable().str();

      std::string offset;

      std::vector<GiNaC::ex> indices;
      for (size_t i = 1; i < subscriptionOp->getNumOperands(); ++i) {
        if (offset.length())
          offset += '_';

        GiNaC::ex expression = valueToExpressionMap[subscriptionOp->getOperand(i)];
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
        if (auto writeOp = mlir::dyn_cast<mlir::modelica::LoadOp>(write); writeOp == loadOp) {
          symbolNameToInfoMap[variableName].matchedEquation = matchedEquation;
        }
      }

      valueToExpressionMap[loadOp->getResult(0)] = symbolNameToInfoMap[variableName].symbol;
    } else {
      matchedEquation->dumpIR();
      subscriptionOp->dump();
      subscriptionOp->getOperand(0).getDefiningOp()->dump();
      llvm_unreachable("Not a VariableGetOp.");
    }
  }
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::ConstantOp constantOp)
{
  mlir::Attribute attribute = constantOp.getValue();

  GiNaC::ex res;

  if (const auto integerValue = attribute.dyn_cast<mlir::IntegerAttr>()) {
    res = integerValue.getInt();
  } else if (const auto modelicaIntegerValue = attribute.dyn_cast<mlir::modelica::IntegerAttr>()) {
    res = modelicaIntegerValue.getValue().getSExtValue();
  } else {
    res = getDoubleFromAttribute(constantOp.getValue());
  }

  valueToExpressionMap[constantOp.getResult()] = res;
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::TimeOp timeOp)
{
  valueToExpressionMap[timeOp.getResult()] = symbolNameToInfoMap["time"].symbol;
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::NegateOp negateOp)
{
  valueToExpressionMap[negateOp.getResult()] = - valueToExpressionMap[negateOp->getOperand(0)];
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::AddOp addOp)
{
  valueToExpressionMap[addOp.getResult()] = valueToExpressionMap[addOp->getOperand(0)] + valueToExpressionMap[addOp->getOperand(1)];
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::SubOp subOp)
{
  valueToExpressionMap[subOp.getResult()] = valueToExpressionMap[subOp->getOperand(0)] - valueToExpressionMap[subOp->getOperand(1)];
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::MulOp mulOp)
{
  valueToExpressionMap[mulOp.getResult()] = valueToExpressionMap[mulOp->getOperand(0)] * valueToExpressionMap[mulOp->getOperand(1)];
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::DivOp divOp)
{
  valueToExpressionMap[divOp.getResult()] = valueToExpressionMap[divOp->getOperand(0)] / valueToExpressionMap[divOp->getOperand(1)];
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::PowOp powOp)
{
  valueToExpressionMap[powOp.getResult()] = GiNaC::pow(valueToExpressionMap[powOp->getOperand(0)], valueToExpressionMap[powOp->getOperand(1)]);
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::SinOp sinOp)
{
  valueToExpressionMap[sinOp.getResult()] = sin(valueToExpressionMap[sinOp->getOperand(0)]);
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::EquationSideOp equationSideOp)
{
  valueToExpressionMap[equationSideOp.getResult()] = valueToExpressionMap[equationSideOp->getOperand(0)];
}

void ModelicaToSymbolicEquationVisitor::visit(mlir::modelica::EquationSidesOp equationSidesOp)
{
  solution = valueToExpressionMap[equationSidesOp->getOperand(0)] == valueToExpressionMap[equationSidesOp->getOperand(1)];
}
