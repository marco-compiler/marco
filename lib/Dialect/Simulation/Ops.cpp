#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "marco/Dialect/Simulation/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace ::mlir::simulation;

#define GET_OP_CLASSES
#include "marco/Dialect/Simulation/Simulation.cpp.inc"

//===---------------------------------------------------------------------===//
// ModuleOp

namespace mlir::simulation
{
  void ModuleOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef modelName,
      llvm::ArrayRef<mlir::Attribute> variables,
      llvm::ArrayRef<mlir::Attribute> derivatives)
  {
    state.addRegion()->emplaceBlock();

    state.addAttribute(
        getModelNameAttrName(state.name),
        builder.getStringAttr(modelName));

    state.addAttribute(
        getVariablesAttrName(state.name),
        builder.getArrayAttr(variables));

    state.addAttribute(
        getDerivativesAttrName(state.name),
        builder.getArrayAttr(derivatives));
  }

  llvm::SmallVector<mlir::Type> ModuleOp::getVariablesTypes()
  {
    llvm::SmallVector<mlir::Type> result;

    for (VariableAttr variable : getVariables().getAsRange<VariableAttr>()) {
      result.push_back(variable.getType());
    }

    return result;
  }

  llvm::StringMap<llvm::StringRef> ModuleOp::getDerivativesMap()
  {
    llvm::StringMap<llvm::StringRef> result;

    for (DerivativeAttr derivativeAttr :
         getDerivatives().getAsRange<DerivativeAttr>()) {
      llvm::StringRef variableName = derivativeAttr.getVariable().getName();

      llvm::StringRef derivativeName =
          derivativeAttr.getDerivative().getName();

      result[variableName] = derivativeName;
    }

    return result;
  }
}

//===---------------------------------------------------------------------===//
// InitFunctionOp

namespace mlir::simulation
{
  mlir::ParseResult InitFunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();
    llvm::SmallVector<mlir::Type> variableTypes;
    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseLParen() ||
        parser.parseRParen() ||
        parser.parseArrow() ||
        parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::Paren,
            [&]() -> mlir::ParseResult {
              mlir::Type type;

              if (mlir::failed(parser.parseType(type))) {
                return mlir::failure();
              }

              variableTypes.push_back(type);
              return mlir::success();
            }) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion)) {
      return mlir::failure();
    }

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(builder.getFunctionType(
            llvm::None, variableTypes)));

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void InitFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " () -> (";

    for (const auto& type : llvm::enumerate(getFunctionType().getResults())) {
      if (type.index() != 0) {
        printer << ", ";
      }

      printer << type.value();
    }

    printer << ")";

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(getFunctionTypeAttrName().getValue());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer << " ";
    printer.printRegion(getBodyRegion(), false);
  }
}

//===---------------------------------------------------------------------===//
// DeinitOp

namespace mlir::simulation
{
  mlir::ParseResult DeinitFunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    llvm::SmallVector<mlir::OpAsmParser::Argument> args;
    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseArgumentList(args, mlir::AsmParser::Delimiter::Paren, true) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion, args)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Type> argTypes;

    for (mlir::OpAsmParser::Argument arg : args) {
      argTypes.push_back(arg.type);
    }

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(builder.getFunctionType(
                argTypes, llvm::None)));

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void DeinitFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << "(";

    for (const auto& arg : llvm::enumerate(getBodyRegion().getArguments())) {
      if (arg.index() != 0) {
        printer << ", ";
      }

      printer.printRegionArgument(arg.value());
    }

    printer << ")";

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(getFunctionTypeAttrName().getValue());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer << " ";
    printer.printRegion(getBodyRegion(), false);
  }

  mlir::ValueRange DeinitFunctionOp::getVariables()
  {
    return getBodyRegion().getArguments();
  }
}

//===---------------------------------------------------------------------===//
// InitICSolversFunctionOp

namespace mlir::simulation
{
  mlir::ParseResult InitICSolversFunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    llvm::SmallVector<mlir::OpAsmParser::Argument> variables;
    llvm::SmallVector<mlir::Type> solverTypes;

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseArgumentList(variables, mlir::AsmParser::Delimiter::Paren, true) ||
        parser.parseArrowTypeList(solverTypes) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion, variables)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Type> variableTypes;

    for (mlir::OpAsmParser::Argument variable : variables) {
      variableTypes.push_back(variable.type);
    }

    auto functionType = builder.getFunctionType(variableTypes, solverTypes);

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(functionType));

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void InitICSolversFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " (";

    for (mlir::BlockArgument variable : getVariables()) {
      if (variable.getArgNumber() != 0) {
        printer << ", ";
      }

      printer.printRegionArgument(variable);
    }

    printer << ")";
    printer.printArrowTypeList(getSolverTypes());
    printer << " ";

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(getFunctionTypeAttrName().getValue());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer.printRegion(getBodyRegion(), false);
  }
}

//===---------------------------------------------------------------------===//
// InitMainSolversFunctionOp

namespace mlir::simulation
{
  mlir::ParseResult InitMainSolversFunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    llvm::SmallVector<mlir::OpAsmParser::Argument> variables;
    llvm::SmallVector<mlir::Type> solverTypes;

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseArgumentList(variables, mlir::AsmParser::Delimiter::Paren, true) ||
        parser.parseArrowTypeList(solverTypes) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion, variables)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Type> variableTypes;

    for (mlir::OpAsmParser::Argument variable : variables) {
      variableTypes.push_back(variable.type);
    }

    auto functionType = builder.getFunctionType(variableTypes, solverTypes);

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(functionType));

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void InitMainSolversFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " (";

    for (mlir::BlockArgument variable : getVariables()) {
      if (variable.getArgNumber() != 0) {
        printer << ", ";
      }

      printer.printRegionArgument(variable);
    }

    printer << ")";
    printer.printArrowTypeList(getSolverTypes());
    printer << " ";

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(getFunctionTypeAttrName().getValue());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer.printRegion(getBodyRegion(), false);
  }
}

//===---------------------------------------------------------------------===//
// DeinitICSolversFunctionOp

namespace mlir::simulation
{
  mlir::ParseResult DeinitICSolversFunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    llvm::SmallVector<mlir::OpAsmParser::Argument> solvers;

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseArgumentList(solvers, mlir::AsmParser::Delimiter::Paren, true) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion, solvers)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Type> solverTypes;

    for (mlir::OpAsmParser::Argument solver : solvers) {
      solverTypes.push_back(solver.type);
    }

    auto functionType = builder.getFunctionType(solverTypes, llvm::None);

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(functionType));

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void DeinitICSolversFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " (";

    for (const auto& solver : llvm::enumerate(getSolvers())) {
      if (solver.index() != 0) {
        printer << ", ";
      }

      printer.printRegionArgument(solver.value());
    }

    printer << ") ";

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(getFunctionTypeAttrName().getValue());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer.printRegion(getBodyRegion(), false);
  }
}

//===---------------------------------------------------------------------===//
// DeinitMainSolversFunctionOp

namespace mlir::simulation
{
  mlir::ParseResult DeinitMainSolversFunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    llvm::SmallVector<mlir::OpAsmParser::Argument> solvers;

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseArgumentList(solvers, mlir::AsmParser::Delimiter::Paren, true) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion, solvers)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Type> solverTypes;

    for (mlir::OpAsmParser::Argument solver : solvers) {
      solverTypes.push_back(solver.type);
    }

    auto functionType = builder.getFunctionType(solverTypes, llvm::None);

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(functionType));

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void DeinitMainSolversFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " (";

    for (const auto& solver : llvm::enumerate(getSolvers())) {
      if (solver.index() != 0) {
        printer << ", ";
      }

      printer.printRegionArgument(solver.value());
    }

    printer << ") ";

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(getFunctionTypeAttrName().getValue());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer.printRegion(getBodyRegion(), false);
  }
}

//===---------------------------------------------------------------------===//
// VariableGetterOp

namespace mlir::simulation
{
  mlir::ParseResult VariableGetterOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();
    llvm::SmallVector<mlir::Attribute> variables;

    if (parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::Square,
            [&]() -> mlir::ParseResult {
              mlir::Attribute currentVariable;

              mlir::ParseResult result =
                  parser.parseAttribute(currentVariable);

              variables.push_back(currentVariable);
              return result;
            })) {
      return mlir::failure();
    }

    result.addAttribute(
        getVariablesAttrName(result.name),
        builder.getArrayAttr(variables));

    llvm::SmallVector<mlir::OpAsmParser::Argument> args;
    mlir::Type resultType;

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseArgumentList(
            args, mlir::AsmParser::Delimiter::Paren, true) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion, args)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Type> argTypes;

    for (mlir::OpAsmParser::Argument arg : args) {
      argTypes.push_back(arg.type);
    }

    auto functionType = builder.getFunctionType(argTypes, resultType);

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(functionType));

    return mlir::success();
  }

  void VariableGetterOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " " << getVariables();
    printer << "(";
    printer.printRegionArgument(getVariable());

    for (mlir::BlockArgument index : getIndices()) {
      printer << ", ";
      printer.printRegionArgument(index);
    }

    printer << ") -> " << getFunctionType().getResults()[0];

    llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
    elidedAttrs.push_back(getFunctionTypeAttrName().getValue());
    elidedAttrs.push_back(getVariablesAttrName().getValue());
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elidedAttrs);

    printer << " ";
    printer.printRegion(getBodyRegion(), false);
  }

  mlir::BlockArgument VariableGetterOp::getVariable()
  {
    return getBodyRegion().getArgument(0);
  }

  int64_t VariableGetterOp::getVariableRank()
  {
    return getIndices().size();
  }

  llvm::ArrayRef<mlir::BlockArgument> VariableGetterOp::getIndices()
  {
    return getBodyRegion().getArguments().drop_front();
  }

  mlir::BlockArgument VariableGetterOp::getIndex(int64_t dimension)
  {
    return getBodyRegion().getArgument(1 + dimension);
  }
}

//===---------------------------------------------------------------------===//
// FunctionOp

namespace mlir::simulation
{
  mlir::ParseResult FunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(
            nameAttr,
            mlir::SymbolTable::getSymbolAttrName(),
            result.attributes)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::OpAsmParser::Argument> solvers;
    mlir::OpAsmParser::Argument time;
    llvm::SmallVector<mlir::OpAsmParser::Argument> variables;
    llvm::SmallVector<mlir::OpAsmParser::Argument> extraArgs;
    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (parser.parseLParen() ||
        parser.parseKeyword("solvers") ||
        parser.parseColon() ||
        parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::Square, [&]() -> mlir::ParseResult {
              mlir::OpAsmParser::Argument arg;

              if (parser.parseArgument(arg, true)) {
                return mlir::failure();
              }

              solvers.push_back(arg);
              return mlir::success();
            }) ||
        parser.parseComma() ||
        parser.parseKeyword("time") ||
        parser.parseColon() ||
        parser.parseLSquare() ||
        parser.parseArgument(time, true) ||
        parser.parseRSquare() ||
        parser.parseComma() ||
        parser.parseKeyword("variables") ||
        parser.parseColon() ||
        parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::Square, [&]() -> mlir::ParseResult {
              mlir::OpAsmParser::Argument arg;

              if (parser.parseArgument(arg, true)) {
                return mlir::failure();
              }

              variables.push_back(arg);
              return mlir::success();
            }) ||
        parser.parseComma() ||
        parser.parseKeyword("extra_args") ||
        parser.parseColon() ||
        parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::Square, [&]() -> mlir::ParseResult {
              mlir::OpAsmParser::Argument arg;

              if (parser.parseArgument(arg, true)) {
                return mlir::failure();
              }

              extraArgs.push_back(arg);
              return mlir::success();
            }) ||
        parser.parseRParen() ||
        parser.parseArrowTypeList(resultTypes) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::OpAsmParser::Argument> args;
    llvm::SmallVector<mlir::Type> argTypes;

    for (mlir::OpAsmParser::Argument solver : solvers) {
      args.push_back(solver);
      argTypes.push_back(solver.type);
    }

    args.push_back(time);
    argTypes.push_back(time.type);

    for (mlir::OpAsmParser::Argument variable : variables) {
      args.push_back(variable);
      argTypes.push_back(variable.type);
    }

    for (mlir::OpAsmParser::Argument extraArg : extraArgs) {
      args.push_back(extraArg);
      argTypes.push_back(extraArg.type);
    }

    auto functionType = builder.getFunctionType(argTypes, resultTypes);

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(functionType));

    result.addAttribute(
        getSolversAmountAttrName(result.name),
        builder.getI64IntegerAttr(solvers.size()));

    result.addAttribute(
        getVariablesAmountAttrName(result.name),
        builder.getI64IntegerAttr(variables.size()));

    mlir::Region* region = result.addRegion();

    if (parser.parseRegion(*region, args)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void FunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());
    printer << "(solvers: [";

    for (const auto& solver : llvm::enumerate(getSolvers())) {
      if (solver.index() != 0) {
        printer << ", ";
      }

      printer.printRegionArgument(solver.value());
    }

    printer << "], time: [";
    printer.printRegionArgument(getTime());
    printer << "], variables: [";

    for (const auto& variable : llvm::enumerate(getVariables())) {
      if (variable.index() != 0) {
        printer << ", ";
      }

      printer.printRegionArgument(variable.value());
    }

    printer << "], extra_args: [";

    for (const auto& extraArg : llvm::enumerate(getExtraArgs())) {
      if (extraArg.index() != 0) {
        printer << ", ";
      }

      printer.printRegionArgument(extraArg.value());
    }

    printer << "])";
    printer.printArrowTypeList(getResultTypes());
    printer << " ";

    llvm::SmallVector<llvm::StringRef, 3> elidedAttrs;
    elidedAttrs.push_back(getSymNameAttrName().getValue());
    elidedAttrs.push_back(getFunctionTypeAttrName().getValue());
    elidedAttrs.push_back(getSolversAmountAttrName().getValue());
    elidedAttrs.push_back(getVariablesAmountAttrName().getValue());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer.printRegion(getBodyRegion(), false);
  }

  void FunctionOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name,
      mlir::TypeRange solverTypes,
      mlir::Type timeType,
      mlir::TypeRange variableTypes,
      mlir::TypeRange extraArgs,
      mlir::TypeRange resultTypes)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));

    llvm::SmallVector<mlir::Type> argTypes;
    argTypes.append(solverTypes.begin(), solverTypes.end());
    argTypes.push_back(timeType);
    argTypes.append(variableTypes.begin(), variableTypes.end());
    argTypes.append(extraArgs.begin(), extraArgs.end());

    auto functionType = builder.getFunctionType(argTypes, resultTypes);

    state.addAttribute(
        getFunctionTypeAttrName(state.name),
        mlir::TypeAttr::get(functionType));

    state.addAttribute(
        getSolversAmountAttrName(state.name),
        builder.getI64IntegerAttr(solverTypes.size()));

    state.addAttribute(
        getVariablesAmountAttrName(state.name),
        builder.getI64IntegerAttr(variableTypes.size()));

    state.addRegion();
  }

  llvm::SmallVector<mlir::Type> FunctionOp::getSolverTypes()
  {
    llvm::SmallVector<mlir::Type> result;

    for (mlir::Value solver : getSolvers()) {
      result.push_back(solver.getType());
    }

    return result;
  }

  mlir::Type FunctionOp::getTimeType()
  {
    return getTime().getType();
  }

  llvm::SmallVector<mlir::Type> FunctionOp::getVariableTypes()
  {
    llvm::SmallVector<mlir::Type> result;

    for (mlir::Value variable : getVariables()) {
      result.push_back(variable.getType());
    }

    return result;
  }

  llvm::SmallVector<mlir::Type> FunctionOp::getExtraArgTypes()
  {
    llvm::SmallVector<mlir::Type> result;

    for (mlir::Value arg : getExtraArgs()) {
      result.push_back(arg.getType());
    }

    return result;
  }

  llvm::ArrayRef<mlir::BlockArgument> FunctionOp::getSolvers()
  {
    return getBodyRegion().getArguments().take_front(getSolversAmount());
  }

  mlir::BlockArgument FunctionOp::getTime()
  {
    return getBodyRegion().getArgument(getSolversAmount());
  }

  llvm::ArrayRef<mlir::BlockArgument> FunctionOp::getVariables()
  {
    return getBodyRegion().getArguments().slice(
        getSolversAmount() + 1, getVariablesAmount());
  }

  llvm::ArrayRef<mlir::BlockArgument> FunctionOp::getExtraArgs()
  {
    return getBodyRegion().getArguments().drop_front(
        getSolversAmount() + 1 + getVariablesAmount());
  }
}
