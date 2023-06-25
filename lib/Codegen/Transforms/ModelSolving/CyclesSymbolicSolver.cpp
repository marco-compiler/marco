#include "marco/Codegen/Transforms/ModelSolving/CyclesSymbolicSolver.h"
#include "ginac/ginac.h"
#include <string>
#include <utility>

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

GiNaC::ex getExpressionFromEquation(MatchedEquation* matchedEquation, std::map<std::string, SymbolInfo>& symbolNameToInfoMap, marco::modeling::IndexSet subscriptionIndices) {
  auto terminator = mlir::dyn_cast<mlir::modelica::EquationSidesOp>(matchedEquation->getOperation().bodyBlock()->getTerminator());

  GiNaC::ex solution;
  ModelicaToSymbolicEquationVisitor visitor(matchedEquation, symbolNameToInfoMap, solution, subscriptionIndices);

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
    } else {
      operation->dump();
      llvm_unreachable("Unsupported operation");
    }

    ++operation;
  }

  return solution;
}

CyclesSymbolicSolver::CyclesSymbolicSolver(mlir::OpBuilder& builder) : builder(builder)
{
}

void printExpressions(GiNaC::ex system) {
  std::cerr << '{' << std::endl;
  for (size_t i = 0; i < system.nops(); ++i) {
    std::cerr << "  " << system.op(i) << std::endl;
  }
  std::cerr << '}' << std::endl << std::endl;
}

bool CyclesSymbolicSolver::solve(const std::vector<MatchedEquationSubscription>& equationSet)
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
    auto terminator = mlir::cast<mlir::modelica::EquationSidesOp>(equation.equation->getOperation().bodyBlock()->getTerminator());

    GiNaC::ex expression = getExpressionFromEquation(equation.equation, symbolNameToInfoMap, equation.solvedIndices);

    expression = expression.expand();

//    std::cerr << '\n' << "Expression: \n" << expression << '\n';
//    std::cerr << "Indices: \n" << equation.solvedIndices << '\n';
//    equation.equation->dumpIR();

//    // If an equation is trivial instead (e.g. x == 1), save it to later substitute it in the other ones.
//    if (GiNaC::is_a<GiNaC::symbol>(expression.lhs()) && GiNaC::is_a<GiNaC::numeric>(expression.rhs())) {
//      trivialEquations.append(expression);
//    } else {
//      systemEquations.append(expression);
//    }

    systemEquations.append(expression);
  }

  // The variables wrt. which the linear system is solved are the matched ones.
  for (const auto& [name, info] : symbolNameToInfoMap) {
    if (info.matchedEquation != nullptr) {
      matchedVariables.append(info.symbol);
    }
  }

  std::cerr << "System equations: \n";
  printExpressions(systemEquations);
  std::cerr << "Matched variables: \n";
  printExpressions(matchedVariables);

  assert(systemEquations.nops() == matchedVariables.nops() && "Number of equations different from number of matched variables.");

  GiNaC::ex solutionEquations;
  try {
    solutionEquations = GiNaC::lsolve(systemEquations, matchedVariables);
  } catch (std::logic_error& e) {
      // The system is not linear so it cannot be solved by the symbolic solver.
    std::cerr << "The system of equations is not linear" << std::endl;
    return false;
  };

  GiNaC::lst newEquations;

  for (const auto& solutionEquation : solutionEquations) {
    bool solutionIsNew = true;
    GiNaC::ex expandedSolutionEquation = solutionEquation.expand();
    newEquations.append(expandedSolutionEquation);
  }

  std::cerr << "Solution: \n";
  printExpressions(solutionEquations);

  GiNaC::lst checkEquations;

  if (newEquations.nops()) {
    for (const GiNaC::ex& expr : newEquations) {
      std::string matchedVariableName;
      if (GiNaC::is_a<GiNaC::symbol>(expr.lhs())) {
        matchedVariableName = GiNaC::ex_to<GiNaC::symbol>(expr.lhs()).get_name();
      } else {
        llvm_unreachable("Expected the left hand side of the solutionEquations equation to be a symbol.");
      }

      SymbolInfo symbolInfo = symbolNameToInfoMap[matchedVariableName];
      auto equation = symbolInfo.matchedEquation;

      if (hasSolvedEquation(equation, symbolInfo.subscriptionIndices)) {
        // If the equation to be inserted has already been solved, then the cycle
        // doesn't exist anymore.
        std::cerr << "Equation already solved, continuing." << std::endl;
        continue;
      }

      auto simpleEquation = Equation::build(equation->cloneIR(), equation->getVariables());
      auto matchedEquation = newEquations_.add(std::make_unique<MatchedEquation>(MatchedEquation(std::move(simpleEquation), symbolInfo.subscriptionIndices, EquationPath::LEFT)));

      auto equationOp = matchedEquation->getOperation();
      auto loc = equationOp.getLoc();
      builder.setInsertionPoint(equationOp);

      equationOp.bodyBlock()->erase();
      assert(equationOp.getBodyRegion().empty());
      mlir::Block* equationBodyBlock = builder.createBlock(&equationOp.getBodyRegion());
      builder.setInsertionPointToStart(equationBodyBlock);

      MatchedEquationSubscription matchedEquationSubscription(matchedEquation, symbolInfo.subscriptionIndices);

      SymbolicToModelicaEquationVisitor visitor = SymbolicToModelicaEquationVisitor(builder, loc, matchedEquationSubscription, symbolNameToInfoMap);

      expr.traverse_postorder(visitor);

      addSolvedEquation(solvedEquations_, equation, symbolInfo.subscriptionIndices);

      GiNaC::ex checkExpression = getExpressionFromEquation(matchedEquation, symbolNameToInfoMap, symbolInfo.subscriptionIndices);
      checkEquations.append(checkExpression);

//      std::cerr << "Indices: " << symbolInfo.subscriptionIndices;
//      std::cerr << "New equation: " << std::endl;
//      matchedEquation->dumpIR();
    }

    std::cerr << "Check: \n";
    printExpressions(checkEquations);



//    for (const auto& equation : newEquations_) {
//      std::cerr << std::endl << "New equation: " << std::endl;
//      equation->dumpIR();
//    }

    return true;
  }

  std::cerr << "The system of equations was already solved." << std::endl;
  return true;
}

SymbolicToModelicaEquationVisitor::SymbolicToModelicaEquationVisitor(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    MatchedEquationSubscription matchedEquation,
    std::map<std::string, SymbolInfo> symbolNameToInfoMap
    ) : builder(builder), loc(loc), matchedEquation(matchedEquation), symbolNameToInfoMap(symbolNameToInfoMap)
{
  std::stack<mlir::modelica::ForEquationOp> forEquations;
  auto forEquationOp = matchedEquation.equation->getOperation()->getParentOfType<mlir::modelica::ForEquationOp>();
  while (forEquationOp) {
    forEquations.push(forEquationOp);
    forEquationOp = forEquationOp->getParentOfType<mlir::modelica::ForEquationOp>();
  }

  size_t index = 0;
  while (!forEquations.empty()) {
    mlir::modelica::ForEquationOp op = forEquations.top();
    forEquations.pop();

    llvm::APInt from = op.getFrom();
    llvm::APInt to = op.getTo();

    matchedEquationForEquationRanges.emplace_back(from, to);

    mlir::BlockArgument blockArgument = op.bodyBlock()->getArgument(0);
    std::string argumentName = "%arg" + std::to_string(index);

    GiNaC::symbol argumentSymbol = symbolNameToInfoMap[argumentName].symbol;
    expressionHashToValueMap[argumentSymbol] = blockArgument;

    ++index;
  }
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::add & x) {
  mlir::Value lhs = expressionHashToValueMap[x.op(0)];

  for (size_t i = 1; i < x.nops(); ++i) {
    mlir::Value rhs = expressionHashToValueMap[x.op(i)];
    mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());
    lhs = builder.create<mlir::modelica::AddOp>(loc, type, lhs, rhs);
  }

  expressionHashToValueMap[x] = lhs;
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::mul & x) {
  mlir::Value lhs = expressionHashToValueMap[x.op(0)];

//  for (auto& [expr, value] : expressionHashToValueMap) {
//    std::cerr << expr << std::endl;
//    value.dump();
//    std::cerr << std::endl;
//  }

  for (size_t i = 1; i < x.nops(); ++i) {
    mlir::Value rhs = expressionHashToValueMap[x.op(i)];
    mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());
    lhs = builder.create<mlir::modelica::MulOp>(loc, type, lhs, rhs);
  }

  expressionHashToValueMap[x] = lhs;
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::power & x) {
  mlir::Value lhs = expressionHashToValueMap[x.op(0)];
  mlir::Value rhs = expressionHashToValueMap[x.op(1)];

  mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());

  mlir::Value value = builder.create<mlir::modelica::PowOp>(loc, type, lhs, rhs);
  expressionHashToValueMap[x] = value;
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::function & x) {
  if (x.get_name() == "sin") {
    mlir::Value lhs = expressionHashToValueMap[x.op(0)];

    mlir::Type type = lhs.getType();

    mlir::Value value = builder.create<mlir::modelica::SinOp>(loc, type, lhs);
    expressionHashToValueMap[x] = value;
  }
}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::relational & x) {
  if (x.info(GiNaC::info_flags::relation_equal)) {
    mlir::Value lhs = expressionHashToValueMap[x.op(0)];
    mlir::Value rhs = expressionHashToValueMap[x.op(1)];

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
  expressionHashToValueMap[x] = value;

}

void SymbolicToModelicaEquationVisitor::visit(const GiNaC::symbol & x) {
  mlir::Value value;
  if (expressionHashToValueMap.count(x) == 0) {
    if (x.get_name() == "time") {
      value = builder.create<mlir::modelica::TimeOp>(loc);
    } else {
      std::vector<mlir::Value> currentIndices;

      std::string variableName = x.get_name();
      SymbolInfo info = symbolNameToInfoMap[variableName];

      // todo: generalize to multiple inductions
      size_t lsb_pos = variableName.find('[', 0);
      int inductionArgument = 0;
//      std::cerr << "Symbol: " << x << std::endl;
      auto indexIterator = *matchedEquation.solvedIndices.begin().operator*().begin();
      while (lsb_pos != std::string::npos) {
        size_t colon_pos = variableName.find(':', lsb_pos);
        long from = std::stoi(variableName.substr(lsb_pos + 1, colon_pos - 1 - lsb_pos));

        size_t rsb_pos = variableName.find(']', colon_pos);
        long to = std::stoi(variableName.substr(colon_pos + 1, rsb_pos -  1 - colon_pos));

        // startIndex should be the index with which the subscripted range begins.
        long startIndex = (*matchedEquation.solvedIndices.begin())[inductionArgument];

//        std::cerr << "Matched indices: " << matchedEquation.solvedIndices << std::endl;
//        std::cerr << "Matched indices: " << (*matchedEquation.solvedIndices.begin())[inductionArgument] << std::endl;

//        for (const auto& [name, infos] : symbolNameToInfoMap) {
//          std::cerr << "Key: " << name << std::endl;
//          std::cerr << "Value: " << infos.symbol << std::endl;
//        }

        mlir::Attribute offset = mlir::modelica::IntegerAttr::get(builder.getContext(), from - startIndex);
        mlir::Value rhs = builder.create<mlir::modelica::ConstantOp>(loc, offset);
        GiNaC::symbol argSymbol = symbolNameToInfoMap["%arg" + std::to_string(inductionArgument)].symbol;

//        std::cerr << "Symbol: " << x << std::endl;
//        std::cerr << "Start index: " << startIndex << std::endl;
//        std::cerr << "from: " << from << std::endl;

//        std::cerr << "Argument symbol: " << argSymbol << std::endl;

//        for (auto& [expression, expValue] : expressionHashToValueMap) {
//          std::cerr << "Key: " << expression << std::endl;
//          std::cerr << "Value: ";
//          expValue.dump();
//          std::cerr << std::endl;
//        }

        mlir::Value lhs = expressionHashToValueMap[argSymbol];
        mlir::Type type = getMostGenericType(lhs.getType(), rhs.getType());
        mlir::Value index = builder.create<mlir::modelica::AddOp>(loc, type, lhs, rhs);

        currentIndices.push_back(index);
        lsb_pos = variableName.find('[', rsb_pos);
        ++inductionArgument;
      }

//      for (const auto& index : info.indices) {
//        if (expressionHashToValueMap.count(index) == 0) {
//          index.traverse_postorder(*this);
//        }
//
//        currentIndices.push_back(expressionHashToValueMap[index]);
//      }

      mlir::Type type = info.variableType;
      std::string baseVariableName = info.variableName;
      if (variableName.rfind("%arg", 0) != std::string::npos) {
        // This variable is an argument, it should already be mapped during visitor initialization.
        value = expressionHashToValueMap[x];
      } else {
        value = builder.create<mlir::modelica::VariableGetOp>(loc, type, baseVariableName);
        if (!currentIndices.empty()) {
          value = builder.create<mlir::modelica::SubscriptionOp>(loc, value, currentIndices);
          value = builder.create<mlir::modelica::LoadOp>(loc, value);
        }
      }
    }

    expressionHashToValueMap[x] = value;
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

ModelicaToSymbolicEquationVisitor::ModelicaToSymbolicEquationVisitor(
    MatchedEquation* matchedEquation,
    std::map<std::string, SymbolInfo>& symbolNameToInfoMap,
    GiNaC::ex& solution, modeling::IndexSet subscriptionIndices
    ) : matchedEquation(matchedEquation), symbolNameToInfoMap(symbolNameToInfoMap), solution(solution), subscriptionIndices(std::move(subscriptionIndices))
{
  // Initialize the block arguments of the valueToExpressionMap

  std::stack<mlir::modelica::ForEquationOp> forEquations;
  auto forEquationOp = matchedEquation->getOperation()->getParentOfType<mlir::modelica::ForEquationOp>();

  while (forEquationOp) {
    forEquations.push(forEquationOp);
    forEquationOp = forEquationOp->getParentOfType<mlir::modelica::ForEquationOp>();
  }

  numberOfForLoops = forEquations.size();

  size_t index = 0;
  while (!forEquations.empty()) {
    mlir::modelica::ForEquationOp op = forEquations.top();
    forEquations.pop();

    mlir::BlockArgument blockArgument = op.bodyBlock()->getArgument(0);
    std::string argumentName = "%arg" + std::to_string(index);

    if (!symbolNameToInfoMap.count(argumentName)) {
      symbolNameToInfoMap[argumentName] = SymbolInfo();
      symbolNameToInfoMap[argumentName].symbol = GiNaC::symbol(argumentName);
      symbolNameToInfoMap[argumentName].variableName = argumentName;
      symbolNameToInfoMap[argumentName].variableType = blockArgument.getType();
      symbolNameToInfoMap[argumentName].indices = {};
    }


    if (!valueToExpressionMap.count(blockArgument)) {
      valueToExpressionMap[blockArgument] = symbolNameToInfoMap[argumentName].symbol;
    }

    ++index;
  }

//  std::cerr << std::endl;
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
      symbolNameToInfoMap[variableName].subscriptionIndices = subscriptionIndices;
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

      // Populate the indices vector with the arguments of the SubscriptionOp
      for (size_t i = 1; i < subscriptionOp->getNumOperands(); ++i) {
        GiNaC::ex index = valueToExpressionMap[subscriptionOp->getOperand(i)];
        indices.push_back(index);

        offset += '[';

        // Assuming there is only one range for now
        // todo: is it possible to have more than one range and how will the code change?
        auto range = *subscriptionIndices.rangesBegin();
//        std::cerr << "Subscription indices: " << subscriptionIndices << std::endl;
        auto iterator = range.begin();
        marco::modeling::Point leftPoint = *iterator;
        marco::modeling::Point rightPoint = leftPoint;
        while (iterator != range.end()) {
          rightPoint = *iterator;
          ++iterator;
        }

        GiNaC::ex leftIndex = index;
        GiNaC::ex rightIndex = index;

        if (leftPoint.rank() == 0) {
          std::cerr << "The rank is zero: scalar variable" << std::endl;
        }

//        for (size_t j = 0; j < leftPoint.rank(); ++j) {
//          std::string blockArgumentName = "%arg" + std::to_string(j);
//          if (symbolNameToInfoMap.count(blockArgumentName)) {
//            GiNaC::symbol argument = symbolNameToInfoMap[blockArgumentName].symbol;
//
//            leftIndex = leftIndex.subs(argument == leftPoint[j]);
//            rightIndex = rightIndex.subs(argument == rightPoint[j]);
//          }
//        }

        std::string blockArgumentName = "%arg" + std::to_string(i - 1);
        if (symbolNameToInfoMap.count(blockArgumentName)) {
          GiNaC::symbol argument = symbolNameToInfoMap[blockArgumentName].symbol;

          leftIndex = leftIndex.subs(argument == leftPoint[i - 1]);
          rightIndex = rightIndex.subs(argument == rightPoint[i - 1]);
        }

        std::ostringstream oss;
        oss << leftIndex;
        oss << ':';
        oss << rightIndex;
        offset += oss.str();

        offset += ']';
      }

      std::string variableName = baseVariableName + offset;

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
          symbolNameToInfoMap[variableName].subscriptionIndices = subscriptionIndices;
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
