#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/IDA/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace ::mlir;
using namespace ::mlir::ida;

#define GET_OP_CLASSES
#include "marco/Dialect/IDA/IDA.cpp.inc"

namespace mlir::ida
{
  //===-------------------------------------------------------------------===//
  // VariableGetterOp
  //===-------------------------------------------------------------------===//

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

  llvm::ArrayRef<BlockArgument> VariableGetterOp::getVariableIndices()
  {
    return getBodyRegion().getArguments();
  }

  //===-------------------------------------------------------------------===//
  // VariableSetterOp
  //===-------------------------------------------------------------------===//

  mlir::ParseResult VariableSetterOp::parse(
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

  void VariableSetterOp::print(OpAsmPrinter &p)
  {
    mlir::function_interface_impl::printFunctionOp(p, *this, false);
  }

  BlockArgument VariableSetterOp::getValue()
  {
    return getBodyRegion().getArgument(0);
  }

  llvm::ArrayRef<BlockArgument> VariableSetterOp::getVariableIndices()
  {
    return getBodyRegion().getArguments().slice(1);
  }

  //===-------------------------------------------------------------------===//
  // AccessFunctionOp
  //===-------------------------------------------------------------------===//

  mlir::ParseResult AccessFunctionOp::parse(
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

  void AccessFunctionOp::print(OpAsmPrinter &p)
  {
    mlir::function_interface_impl::printFunctionOp(p, *this, false);
  }

  llvm::ArrayRef<BlockArgument> AccessFunctionOp::getEquationIndices()
  {
    return getBodyRegion().getArguments();
  }

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
        parser, result, false, buildFuncType);
  }

  void ResidualFunctionOp::print(OpAsmPrinter &p)
  {
    mlir::function_interface_impl::printFunctionOp(p, *this, false);
  }

  mlir::BlockArgument ResidualFunctionOp::getTime()
  {
    return getBodyRegion().getArgument(0);
  }

  llvm::ArrayRef<BlockArgument> ResidualFunctionOp::getEquationIndices()
  {
    return getBodyRegion().getArguments().drop_front(1);
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
        parser, result, false, buildFuncType);
  }

  void JacobianFunctionOp::print(OpAsmPrinter &p)
  {
    mlir::function_interface_impl::printFunctionOp(p, *this, false);
  }

  mlir::BlockArgument JacobianFunctionOp::getTime()
  {
    return getBodyRegion().getArgument(0);
  }

  llvm::ArrayRef<BlockArgument> JacobianFunctionOp::getEquationIndices()
  {
    return getBodyRegion().getArguments().slice(
        1, getEquationRank().getSExtValue());
  }

  llvm::ArrayRef<BlockArgument> JacobianFunctionOp::getVariableIndices()
  {
    size_t offset = getBodyRegion().getNumArguments() -
        getVariableRank().getSExtValue() - 1;

    return getBodyRegion().getArguments().slice(
        offset, getVariableRank().getSExtValue());
  }

  BlockArgument JacobianFunctionOp::getAlpha()
  {
    return getBodyRegion().getArguments().back();
  }
}
