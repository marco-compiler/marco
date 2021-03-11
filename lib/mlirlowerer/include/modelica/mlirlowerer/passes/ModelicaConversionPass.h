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
	struct ModelicaConversionOptions
	{

		/**
		 * Get a statically allocated copy of the default options.
		 *
		 * @return default options
		 */
		static const ModelicaConversionOptions& getDefaultOptions() {
			static ModelicaConversionOptions options;
			return options;
		}
	};

	/**
	 * Pass to convert Modelica operations to a mix of Std, SCF and LLVM ones.
	 *
	 * @param patterns patterns list to populate
	 * @param context	 MLIR context
	 */
	class ModelicaConversionPass: public mlir::PassWrapper<ModelicaConversionPass, mlir::OperationPass<mlir::ModuleOp>> {
		public:
		ModelicaConversionPass(ModelicaConversionOptions options = ModelicaConversionOptions::getDefaultOptions());

		void getDependentDialects(mlir::DialectRegistry &registry) const override;
		void runOnOperation() final;

		private:
		ModelicaConversionOptions options;
	};

	void populateModelicaConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter);

	std::unique_ptr<mlir::Pass> createModelicaConversionPass(ModelicaConversionOptions options = ModelicaConversionOptions::getDefaultOptions());
}
