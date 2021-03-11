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
	class LLVMLoweringPass : public mlir::PassWrapper<LLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
		public:
		LLVMLoweringPass();

		mlir::LogicalResult stdToLLVMConversionPass(mlir::ModuleOp module);
		mlir::LogicalResult castsFolderPass(mlir::ModuleOp module);

		void runOnOperation() final;
	};

	std::unique_ptr<mlir::Pass> createLLVMLoweringPass();
}
