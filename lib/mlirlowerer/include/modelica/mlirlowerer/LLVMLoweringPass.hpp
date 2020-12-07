#include <mlir/IR/Dialect.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

class LLVMLoweringPass : public mlir::PassWrapper<LLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
	public:
	void runOnOperation() final;
};