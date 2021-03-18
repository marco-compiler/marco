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
	void populateModelicaBasicConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaMemoryConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaControlFlowConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaLogicConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaMathConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);
	void populateModelicaBuiltInConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);

	void populateModelicaConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);

	/**
	 * Create a pass to convert Modelica operations to a mix of Std,
	 * SCF and LLVM ones.
 */
	std::unique_ptr<mlir::Pass> createModelicaConversionPass();
}
