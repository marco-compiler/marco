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

	class ModelicaFinalizerPass : public mlir::PassWrapper<ModelicaFinalizerPass, mlir::OperationPass<mlir::ModuleOp>> {
		public:
		ModelicaFinalizerPass();

		//mlir::LogicalResult step1(mlir::ModuleOp module);

		void runOnOperation() final;
	};

	void populateModelicaFinalizerPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, mlir::TypeConverter& typeConverter);

	std::unique_ptr<mlir::Pass> createModelicaFinalizerPass();
}
