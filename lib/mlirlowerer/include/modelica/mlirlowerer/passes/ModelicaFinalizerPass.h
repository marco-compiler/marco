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

		mlir::LogicalResult stdToLLVMConversionPass(mlir::ModuleOp module);
		mlir::LogicalResult castsFolderPass(mlir::ModuleOp module);

		void runOnOperation() final;
	};

	std::unique_ptr<mlir::Pass> createModelicaFinalizerPass();
}
