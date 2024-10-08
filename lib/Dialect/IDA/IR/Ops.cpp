#include "marco/Dialect/IDA/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces//FunctionImplementation.h"

using namespace ::mlir;
using namespace ::mlir::ida;

#define GET_OP_CLASSES
#include "marco/Dialect/IDA/IR/IDAOps.cpp.inc"

namespace mlir::ida {
//===-------------------------------------------------------------------===//
// ResidualFunctionOp
//===-------------------------------------------------------------------===//

mlir::ParseResult ResidualFunctionOp::parse(mlir::OpAsmParser &parser,
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

void ResidualFunctionOp::print(mlir::OpAsmPrinter &printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

mlir::BlockArgument ResidualFunctionOp::getTime() {
  return getBody().getArgument(0);
}

llvm::ArrayRef<mlir::BlockArgument> ResidualFunctionOp::getEquationIndices() {
  return getBody().getArguments().drop_front(1);
}

//===-------------------------------------------------------------------===//
// JacobianFunctionOp
//===-------------------------------------------------------------------===//

mlir::ParseResult JacobianFunctionOp::parse(mlir::OpAsmParser &parser,
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

void JacobianFunctionOp::print(mlir::OpAsmPrinter &printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

mlir::BlockArgument JacobianFunctionOp::getTime() {
  return getBody().getArgument(0);
}

llvm::ArrayRef<mlir::BlockArgument> JacobianFunctionOp::getEquationIndices() {
  return getBody().getArguments().slice(1, getEquationRank().getSExtValue());
}

llvm::ArrayRef<mlir::BlockArgument> JacobianFunctionOp::getVariableIndices() {
  int64_t equationRank = getEquationRank().getSExtValue();
  int64_t variableRank = getVariableRank().getSExtValue();
  int64_t offset = 1 + equationRank;
  return getBody().getArguments().slice(offset, variableRank);
}

mlir::BlockArgument JacobianFunctionOp::getAlpha() {
  int64_t equationRank = getEquationRank().getSExtValue();
  int64_t variableRank = getVariableRank().getSExtValue();
  int64_t offset = 1 + equationRank + variableRank;

  return getBody().getArguments()[offset];
}

mlir::BlockArgument JacobianFunctionOp::getMemoryPool() {
  int64_t equationRank = getEquationRank().getSExtValue();
  int64_t variableRank = getVariableRank().getSExtValue();
  int64_t offset = 1 + equationRank + variableRank + 1;
  return getBody().getArguments()[offset];
}

llvm::ArrayRef<mlir::BlockArgument> JacobianFunctionOp::getADSeeds() {
  int64_t equationRank = getEquationRank().getSExtValue();
  int64_t variableRank = getVariableRank().getSExtValue();
  int64_t offset = 1 + equationRank + variableRank + 2;
  return getBody().getArguments().slice(offset);
}
} // namespace mlir::ida
