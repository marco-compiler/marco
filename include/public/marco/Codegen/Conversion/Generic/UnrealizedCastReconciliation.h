#ifndef MARCO_UNREALIZEDCASTRECONCILIATION_H
#define MARCO_UNREALIZEDCASTRECONCILIATION_H

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createUnrealizedCastReconciliationPass();
}

#endif//MARCO_UNREALIZEDCASTRECONCILIATION_H
