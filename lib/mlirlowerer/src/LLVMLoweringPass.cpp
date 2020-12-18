#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Target/LLVMIR.h>
#include <modelica/mlirlowerer/LLVMLoweringPass.hpp>

using namespace mlir;

void LLVMLoweringPass::runOnOperation() {
	LLVMConversionTarget target(getContext());
	target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

	LLVMTypeConverter typeConverter(&getContext());

	OwningRewritePatternList patterns;
	populateLoopToStdConversionPatterns(patterns, &getContext());
	populateStdToLLVMConversionPatterns(typeConverter, patterns);

	auto module = getOperation();

	if (failed(applyFullConversion(module, target, std::move(patterns))))
		signalPassFailure();
}
