#include "marco/Dialect/SUNDIALS/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace ::mlir;
using namespace ::mlir::sundials;

#define GET_OP_CLASSES
#include "marco/Dialect/SUNDIALS/IR/SUNDIALSOps.cpp.inc"

namespace mlir::sundials {
//===-------------------------------------------------------------------===//
// VariableGetterOp
//===-------------------------------------------------------------------===//

mlir::ParseResult VariableGetterOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name),
      buildFuncType, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

void VariableGetterOp::print(mlir::OpAsmPrinter &printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

llvm::ArrayRef<mlir::BlockArgument> VariableGetterOp::getVariableIndices() {
  return getBody().getArguments();
}

//===-------------------------------------------------------------------===//
// VariableSetterOp
//===-------------------------------------------------------------------===//

mlir::ParseResult VariableSetterOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name),
      buildFuncType, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

void VariableSetterOp::print(mlir::OpAsmPrinter &printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

mlir::BlockArgument VariableSetterOp::getValue() {
  return getBody().getArgument(0);
}

llvm::ArrayRef<mlir::BlockArgument> VariableSetterOp::getVariableIndices() {
  return getBody().getArguments().slice(1);
}

//===-------------------------------------------------------------------===//
// AccessFunctionOp
//===-------------------------------------------------------------------===//

mlir::ParseResult AccessFunctionOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name),
      buildFuncType, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

void AccessFunctionOp::print(mlir::OpAsmPrinter &printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

llvm::ArrayRef<mlir::BlockArgument> AccessFunctionOp::getEquationIndices() {
  return getBody().getArguments();
}
} // namespace mlir::sundials
