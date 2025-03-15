#include "marco/Dialect/Runtime/IR/Ops.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace ::mlir::runtime;

static bool parsePrintableIndicesList(mlir::OpAsmParser &parser,
                                      PrintableIndicesList &prop) {
  if (parser.parseLSquare()) {
    return true;
  }

  if (mlir::failed(parser.parseOptionalRSquare())) {
    do {
      if (mlir::succeeded(parser.parseOptionalKeyword("true"))) {
        prop.emplace_back(true);
      } else if (mlir::succeeded(parser.parseOptionalKeyword("false"))) {
        prop.emplace_back(false);
      } else {
        IndexSet indices;

        if (mlir::failed(mlir::modeling::parse(parser, indices))) {
          return true;
        }

        prop.emplace_back(std::move(indices));
      }
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare()) {
      return true;
    }
  }

  return false;
}

static void printPrintableIndicesList(mlir::OpAsmPrinter &printer,
                                      mlir::Operation *op,
                                      const PrintableIndicesList &prop) {
  printer << "[";

  llvm::interleaveComma(prop, printer, [&](const PrintInfo &printInfo) {
    if (printInfo.isa<bool>()) {
      printer << (printInfo.get<bool>() ? "true" : "false");
      return;
    }

    mlir::modeling::print(printer, printInfo.get<IndexSet>());
  });

  printer << "]";
}

#define GET_OP_CLASSES
#include "marco/Dialect/Runtime/IR/RuntimeOps.cpp.inc"

//===---------------------------------------------------------------------===//
// InitFunctionOp

namespace mlir::runtime {
mlir::ParseResult InitFunctionOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::Region *bodyRegion = result.addRegion();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  result.addAttribute(
      getFunctionTypeAttrName(result.name),
      mlir::TypeAttr::get(builder.getFunctionType(std::nullopt, std::nullopt)));

  if (bodyRegion->empty()) {
    bodyRegion->emplaceBlock();
  }

  return mlir::success();
}

void InitFunctionOp::print(mlir::OpAsmPrinter &printer) {
  llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
  elidedAttrs.push_back(getFunctionTypeAttrName().getValue());

  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           elidedAttrs);

  printer << " ";
  printer.printRegion(getBodyRegion(), false);
}
} // namespace mlir::runtime

//===---------------------------------------------------------------------===//
// DeinitOp

namespace mlir::runtime {
mlir::ParseResult DeinitFunctionOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::Region *bodyRegion = result.addRegion();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  result.addAttribute(
      getFunctionTypeAttrName(result.name),
      mlir::TypeAttr::get(builder.getFunctionType(std::nullopt, std::nullopt)));

  if (bodyRegion->empty()) {
    bodyRegion->emplaceBlock();
  }

  return mlir::success();
}

void DeinitFunctionOp::print(mlir::OpAsmPrinter &printer) {
  llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
  elidedAttrs.push_back(getFunctionTypeAttrName().getValue());

  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           elidedAttrs);

  printer << " ";
  printer.printRegion(getBodyRegion(), false);
}
} // namespace mlir::runtime

//===---------------------------------------------------------------------===//
// VariableGetterOp

namespace mlir::runtime {
void VariableGetterOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, llvm::StringRef name,
                             uint64_t variableRank) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));

  llvm::SmallVector<mlir::Type, 3> argTypes(variableRank,
                                            builder.getIndexType());

  state.addAttribute(getFunctionTypeAttrName(state.name),
                     mlir::TypeAttr::get(builder.getFunctionType(
                         argTypes, builder.getF64Type())));

  state.addRegion();
}

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

mlir::BlockArgument VariableGetterOp::getVariable() {
  return getBodyRegion().getArgument(0);
}

uint64_t VariableGetterOp::getVariableRank() { return getIndices().size(); }

llvm::ArrayRef<mlir::BlockArgument> VariableGetterOp::getIndices() {
  return getBodyRegion().getArguments();
}

mlir::BlockArgument VariableGetterOp::getIndex(uint64_t dimension) {
  return getBodyRegion().getArgument(dimension);
}
} // namespace mlir::runtime

//===---------------------------------------------------------------------===//
// EquationFunctionOp

namespace mlir::runtime {
mlir::ParseResult EquationFunctionOp::parse(mlir::OpAsmParser &parser,
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

void EquationFunctionOp::print(OpAsmPrinter &printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

void EquationFunctionOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &state,
                               llvm::StringRef name, uint64_t numOfInductions,
                               llvm::ArrayRef<mlir::NamedAttribute> attrs,
                               llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));

  llvm::SmallVector<mlir::Type> argTypes(numOfInductions * 2,
                                         builder.getIndexType());

  auto functionType = builder.getFunctionType(argTypes, std::nullopt);

  state.addAttribute(getFunctionTypeAttrName(state.name),
                     mlir::TypeAttr::get(functionType));

  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty()) {
    return;
  }

  assert(functionType.getNumInputs() == argAttrs.size());

  mlir::call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, std::nullopt, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name));
}
} // namespace mlir::runtime

//===---------------------------------------------------------------------===//
// FunctionOp

namespace mlir::runtime {
void FunctionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, mlir::FunctionType functionType) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));

  state.addAttribute(getFunctionTypeAttrName(state.name),
                     mlir::TypeAttr::get(functionType));

  state.addRegion();
}

mlir::ParseResult FunctionOp::parse(mlir::OpAsmParser &parser,
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

void FunctionOp::print(mlir::OpAsmPrinter &printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}
} // namespace mlir::runtime

//===---------------------------------------------------------------------===//
// CallOp

namespace mlir::runtime {
void CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   FunctionOp callee, mlir::ValueRange args) {
  build(builder, state, callee.getResultTypes(), callee.getSymName(), args,
        nullptr, nullptr);
}

mlir::LogicalResult CallOp::verify() {
  if (llvm::any_of(getResultTypes(), [](mlir::Type type) {
        return mlir::isa<mlir::ShapedType>(type);
      })) {
    return emitOpError() << "results must be scalar values";
  }

  return mlir::success();
}

mlir::LogicalResult
CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

  auto functionOp =
      symbolTable.lookupSymbolIn<FunctionOp>(moduleOp, getCalleeAttr());

  if (!functionOp) {
    return emitOpError() << "'" << getCallee()
                         << "' does not reference a valid function";
  }

  mlir::FunctionType functionType = functionOp.getFunctionType();

  if (getArgs().getTypes() != functionType.getInputs()) {
    return emitOpError() << "arguments don't match the function signature";
  }

  if (getResultTypes() != functionType.getResults()) {
    return emitOpError() << "results don't match the function signature";
  }

  return mlir::success();
}
} // namespace mlir::runtime
