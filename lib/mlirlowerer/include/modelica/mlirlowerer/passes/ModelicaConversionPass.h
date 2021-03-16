#pragma once

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>

namespace modelica
{
	/**
	 * Pass to convert Modelica operations to a mix of Std, SCF and LLVM ones.
	 *
	 * @param patterns patterns list to populate
	 * @param context	 MLIR context
	 */
	class ModelicaConversionPass: public mlir::PassWrapper<ModelicaConversionPass, mlir::OperationPass<mlir::ModuleOp>> {
		public:
		void getDependentDialects(mlir::DialectRegistry &registry) const override;
		void runOnOperation() final;
	};

	void populateModelicaBasicConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaMemoryConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaControlFlowConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaLogicConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaMathConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaBuiltInConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);


	void populateModelicaConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);

	std::unique_ptr<mlir::Pass> createModelicaConversionPass();
}
