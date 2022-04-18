#ifndef MARCO_UNREALIZEDCASTSELIMINATION_H
#define MARCO_UNREALIZEDCASTSELIMINATION_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createUnrealizedCastsEliminationPass();

  inline void registerUnrealizedCastsEliminationPass()
  {
    mlir::registerPass(
        "remove-unrealized-casts", "Remove the unrealized casts",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createUnrealizedCastsEliminationPass();
        });
  }
}

#endif//MARCO_UNREALIZEDCASTSELIMINATION_H
