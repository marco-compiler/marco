#ifndef MARCO_CODEGEN_CONVERSION_RUNTIMECOMMON_MANGLING_H
#define MARCO_CODEGEN_CONVERSION_RUNTIMECOMMON_MANGLING_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::runtime {
class RuntimeFunctionsMangling {
public:
  std::string
  getMangledFunction(llvm::StringRef baseName,
                     llvm::StringRef mangledReturnType,
                     llvm::ArrayRef<std::string> argsMangledTypes) const;

  std::string getVoidType() const;

  std::string getIntegerType(unsigned int bitWidth) const;

  std::string getFloatingPointType(unsigned int bitWidth) const;

  std::string getArrayType(llvm::StringRef mangledElementType) const;

  std::string getPointerType(llvm::StringRef mangledElementType) const;

  std::string getVoidPointerType() const;
};
} // namespace mlir::runtime

#endif // MARCO_CODEGEN_CONVERSION_RUNTIMECOMMON_MANGLING_H
