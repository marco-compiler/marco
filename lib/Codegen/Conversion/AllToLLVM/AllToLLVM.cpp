#include "marco/Codegen/Conversion/AllToLLVM/AllToLLVM.h"

#include "marco/Codegen/Conversion/BaseModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/BaseModelicaToLLVM/BaseModelicaToLLVM.h"
#include "marco/Codegen/Conversion/IDACommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/IDAToLLVM/IDAToLLVM.h"
#include "marco/Codegen/Conversion/KINSOLCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/KINSOLToLLVM/KINSOLToLLVM.h"
#include "marco/Codegen/Conversion/RuntimeToLLVM/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/RuntimeToLLVM/RuntimeToLLVM.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_ALLTOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

namespace {
class AllToLLVMConversionPass
    : public mlir::impl::AllToLLVMConversionPassBase<AllToLLVMConversionPass> {
public:
  using AllToLLVMConversionPassBase::AllToLLVMConversionPassBase;

  void runOnOperation() override {
    if (mlir::failed(runConversion())) {
      mlir::emitError(getOperation()->getLoc())
          << "Error in converting all dialects to LLVM";

      return signalPassFailure();
    }
  }

private:
  mlir::LogicalResult runConversion() {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    const auto &dlAnalysis =
        getAnalysisManager().getAnalysis<mlir::DataLayoutAnalysis>();

    // Add all the type conversions.
    mlir::LLVMTypeConverter typeConverter(&getContext(), &dlAnalysis);

    mlir::bmodelica::LLVMTypeConverter bmodelicaTypeConverter(
        &getContext(), dlAnalysis.getAtOrAbove(getOperation()),
        typeConverter.getOptions());

    mlir::ida::LLVMTypeConverter idaTypeConverter(&getContext(),
                                                  typeConverter.getOptions());

    mlir::kinsol::LLVMTypeConverter kinsolTypeConverter(
        &getContext(), typeConverter.getOptions());

    mlir::runtime::LLVMTypeConverter runtimeTypeConverter(
        &getContext(), typeConverter.getOptions());

    typeConverter.addConversion([&](mlir::Type type)
                                    -> std::optional<mlir::Type> {
      if (mlir::isa<mlir::bmodelica::BooleanType, mlir::bmodelica::IntegerType,
                    mlir::bmodelica::RealType, mlir::bmodelica::ArrayType,
                    mlir::bmodelica::UnrankedArrayType,
                    mlir::bmodelica::RangeType>(type)) {
        return bmodelicaTypeConverter.convertType(type);
      }

      if (mlir::isa<mlir::ida::InstanceType, mlir::ida::EquationType,
                    mlir::ida::VariableType>(type)) {
        return idaTypeConverter.convertType(type);
      }

      if (mlir::isa<mlir::kinsol::InstanceType, mlir::kinsol::EquationType,
                    mlir::kinsol::VariableType>(type)) {
        return kinsolTypeConverter.convertType(type);
      }

      if (mlir::isa<mlir::runtime::StringType>(type)) {
        return runtimeTypeConverter.convertType(type);
      }

      return std::nullopt;
    });

    // Caching data structures.
    mlir::SymbolTableCollection symbolTables;
    mlir::kinsol_to_llvm::GlobalsCache kinsolToLLVMGlobalsCache;

    // Collect the conversion patterns.
    mlir::RewritePatternSet patterns(&getContext());
    populateBaseModelicaToLLVMConversionPatterns(patterns, typeConverter,
                                                 symbolTables);

    populateIDAToLLVMConversionPatterns(patterns, typeConverter, symbolTables);

    populateKINSOLToLLVMConversionPatterns(
        patterns, typeConverter, symbolTables, &kinsolToLLVMGlobalsCache);

    populateRuntimeToLLVMPatterns(patterns, typeConverter, symbolTables);

    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    mlir::cf::populateAssertToLLVMConversionPattern(typeConverter, patterns,
                                                    true, &symbolTables);

    mlir::populateComplexToLLVMConversionPatterns(typeConverter, patterns);

    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns,
                                               &symbolTables);

    mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    mlir::populateFinalizeMemRefToLLVMConversionPatterns(
        typeConverter, patterns, &symbolTables);

    // Run the conversion.
    mlir::ConversionConfig config;
    config.allowPatternRollback = false;

    return applyPartialConversion(getOperation(), target, std::move(patterns),
                                  config);
  }
};
} // namespace

namespace mlir {
std::unique_ptr<mlir::Pass> createAllToLLVMConversionPass() {
  return std::make_unique<AllToLLVMConversionPass>();
}
} // namespace mlir
