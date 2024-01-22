#include "marco/Dialect/KINSOL/Ops.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir;
using namespace ::mlir::kinsol;

#define GET_OP_CLASSES
#include "marco/Dialect/KINSOL/KINSOL.cpp.inc"

namespace mlir::kinsol
{
  //===-------------------------------------------------------------------===//
  // ResidualFunctionOp
  //===-------------------------------------------------------------------===//

  mlir::ParseResult ResidualFunctionOp::parse(
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
        parser, result, false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
  }

  void ResidualFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    mlir::function_interface_impl::printFunctionOp(
        printer, *this, false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
  }

  llvm::ArrayRef<mlir::BlockArgument> ResidualFunctionOp::getEquationIndices()
  {
    return getBody().getArguments();
  }

  //===-------------------------------------------------------------------===//
  // JacobianFunctionOp
  //===-------------------------------------------------------------------===//

  mlir::ParseResult JacobianFunctionOp::parse(
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
        parser, result, false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
  }

  void JacobianFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    mlir::function_interface_impl::printFunctionOp(
        printer, *this, false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
  }

  llvm::ArrayRef<mlir::BlockArgument> JacobianFunctionOp::getEquationIndices()
  {
    auto equationRank = getEquationRank().getSExtValue();
    return getBody().getArguments().slice(0, equationRank);
  }

  llvm::ArrayRef<mlir::BlockArgument> JacobianFunctionOp::getVariableIndices()
  {
    auto equationRank = getEquationRank().getSExtValue();
    auto variableRank = getVariableRank().getSExtValue();
    return getBody().getArguments().slice(equationRank, variableRank);
  }
}
