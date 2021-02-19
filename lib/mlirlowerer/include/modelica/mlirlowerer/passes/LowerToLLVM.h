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
	struct ModelicaToLLVMLoweringOptions {

		/**
		 * Get a statically allocated copy of the default options.
		 *
		 * @return default options
		 */
		static const ModelicaToLLVMLoweringOptions& getDefaultOptions() {
			static ModelicaToLLVMLoweringOptions options;
			return options;
		}
	};

	class ModelicaToLLVMLoweringPass : public mlir::PassWrapper<ModelicaToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
		public:
		ModelicaToLLVMLoweringPass(ModelicaToLLVMLoweringOptions options = ModelicaToLLVMLoweringOptions::getDefaultOptions());

		void getDependentDialects(mlir::DialectRegistry &registry) const override;
		void runOnOperation() final;

		private:
		ModelicaToLLVMLoweringOptions options;
	};

	/**
	 * Collect a set of patterns to lower from Modelica operations to operations
	 * within the LLVM dialect.
	 *
	 * @param patterns patterns list to populate
	 * @param context	 MLIR context
	 */
	void populateModelicaToLLVMConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);

	/**
	 * Create a pass to convert Modelica operations to LLVM ones.
	 *
	 * @return pass
	 */
	std::unique_ptr<mlir::Pass> createModelicaToLLVMLoweringPass(ModelicaToLLVMLoweringOptions options = ModelicaToLLVMLoweringOptions::getDefaultOptions());
}
