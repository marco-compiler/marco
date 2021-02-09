#pragma once

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.h>

namespace modelica
{
	class ModelicaLoweringPass : public mlir::PassWrapper<ModelicaLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
		public:
		void getDependentDialects(mlir::DialectRegistry &registry) const override;
		void runOnOperation() final;
	};

	/**
	 * Collect a set of patterns to lower from Modelica operations to operations
	 * within the MLIR dialects.
	 *
	 * @param patterns patterns list to populate
	 * @param context	 MLIR context
	 */
	void populateModelicaConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context);

	/**
	 * Create a pass to convert Modelica operations to Standard and Linalg ones.
	 *
	 * @return pass
	 */
	std::unique_ptr<mlir::Pass> createModelicaLoweringPass();
}
