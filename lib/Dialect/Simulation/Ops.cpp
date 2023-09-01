#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "marco/Dialect/Simulation/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace ::mlir::simulation;

#define GET_OP_CLASSES
#include "marco/Dialect/Simulation/Simulation.cpp.inc"

//===---------------------------------------------------------------------===//
// InitFunctionOp

namespace mlir::simulation
{
  mlir::ParseResult InitFunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();
    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion)) {
      return mlir::failure();
    }

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(builder.getFunctionType(
            llvm::None, llvm::None)));

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void InitFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
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
    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion)) {
      return mlir::failure();
    }

    result.addAttribute(
        getFunctionTypeAttrName(result.name),
        mlir::TypeAttr::get(builder.getFunctionType(
            llvm::None, llvm::None)));

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void DeinitFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(getFunctionTypeAttrName().getValue());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer << " ";
    printer.printRegion(getBodyRegion(), false);
  }
}

//===---------------------------------------------------------------------===//
// VariableGetterOp

namespace mlir::simulation
{
  void VariableGetterOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name,
      uint64_t variableRank)
  {
    state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));

    llvm::SmallVector<mlir::Type, 3> argTypes(
        variableRank, builder.getIndexType());

    state.addAttribute(
        mlir::FunctionOpInterface::getTypeAttrName(),
        mlir::TypeAttr::get(builder.getFunctionType(
            argTypes, builder.getF64Type())));

    state.addRegion();
  }

  mlir::ParseResult VariableGetterOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto buildFuncType =
        [](mlir::Builder& builder,
           llvm::ArrayRef<mlir::Type> argTypes,
           llvm::ArrayRef<mlir::Type> results,
           mlir::function_interface_impl::VariadicFlag,
           std::string&) {
          return builder.getFunctionType(argTypes, results);
        };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, false, buildFuncType);
  }

  void VariableGetterOp::print(OpAsmPrinter &p)
  {
    mlir::function_interface_impl::printFunctionOp(p, *this, false);
  }

  mlir::BlockArgument VariableGetterOp::getVariable()
  {
    return getBodyRegion().getArgument(0);
  }

  uint64_t VariableGetterOp::getVariableRank()
  {
    return getIndices().size();
  }

  llvm::ArrayRef<mlir::BlockArgument> VariableGetterOp::getIndices()
  {
    return getBodyRegion().getArguments();
  }

  mlir::BlockArgument VariableGetterOp::getIndex(uint64_t dimension)
  {
    return getBodyRegion().getArgument(dimension);
  }
}

//===---------------------------------------------------------------------===//
// FunctionOp

namespace mlir::simulation
{
  void FunctionOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name,
      mlir::FunctionType functionType)
  {
    state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));

    state.addAttribute(
        mlir::FunctionOpInterface::getTypeAttrName(),
        mlir::TypeAttr::get(functionType));

    state.addRegion();
  }

  mlir::ParseResult FunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto buildFuncType =
        [](mlir::Builder& builder,
           llvm::ArrayRef<mlir::Type> argTypes,
           llvm::ArrayRef<mlir::Type> results,
           mlir::function_interface_impl::VariadicFlag,
           std::string&) {
          return builder.getFunctionType(argTypes, results);
        };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, false, buildFuncType);
  }

  void FunctionOp::print(OpAsmPrinter &p)
  {
    mlir::function_interface_impl::printFunctionOp(p, *this, false);
  }
}
