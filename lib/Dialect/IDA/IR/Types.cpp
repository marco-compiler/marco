#include "marco/Dialect/IDA/IR/Types.h"
#include "marco/Dialect/IDA/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::ida;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/IDA/IR/Types.cpp.inc"

//===---------------------------------------------------------------------===//
// VariableType
//===---------------------------------------------------------------------===//

// IDA variables are identified by 64-bit integers.

llvm::TypeSize
VariableType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                                mlir::DataLayoutEntryListRef params) const {
  return llvm::TypeSize::getFixed(64);
}

uint64_t
VariableType::getABIAlignment(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const {
  return 8;
}
