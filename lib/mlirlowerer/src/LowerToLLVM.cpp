#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <modelica/mlirlowerer/LowerToLLVM.hpp>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>

using namespace mlir;
using namespace modelica;
using namespace std;

ModelicaToLLVMLoweringPass::ModelicaToLLVMLoweringPass(ModelicaToLLVMLoweringOptions options)
		: options(move(options))
{
}

void ModelicaToLLVMLoweringPass::runOnOperation()
{
	auto module = getOperation();

	ConversionTarget target(getContext());
	target.addLegalDialect<LLVM::LLVMDialect>();
	target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

	// During this lowering, we will also be lowering the MemRef types, that are
	// currently being operated on, to a representation in LLVM. To perform this
	// conversion we use a TypeConverter as part of the lowering. This converter
	// details how one type maps to another. This is necessary now that we will be
	// doing more complicated lowerings, involving loop region arguments.
	LowerToLLVMOptions llvmLoweringOptions;
	llvmLoweringOptions.useBarePtrCallConv = options.useBarePtrCallConv;

	LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions);

	// Provide the set of patterns that will lower the Modelica operations
	mlir::OwningRewritePatternList patterns;
	mlir::vector::populateVectorContractLoweringPatterns(patterns, &getContext());
	populateVectorToLLVMConversionPatterns(typeConverter, patterns);
	populateStdToLLVMConversionPatterns(typeConverter, patterns);

	// With the target and rewrite patterns defined, we can now attempt the
	// conversion. The conversion will signal failure if any of our "illegal"
	// operations were not converted successfully.
	module.dump();

	if (failed(applyFullConversion(module, target, move(patterns))))
		signalPassFailure();

	module.dump();
}

std::unique_ptr<mlir::Pass> modelica::createModelicaToLLVMLoweringPass(ModelicaToLLVMLoweringOptions options)
{
	return std::make_unique<ModelicaToLLVMLoweringPass>(options);
}
