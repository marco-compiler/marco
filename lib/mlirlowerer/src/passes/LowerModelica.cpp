#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <modelica/mlirlowerer/passes/ConversionPatterns.h>
#include <modelica/mlirlowerer/passes/LowerModelica.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace mlir;
using namespace modelica;
using namespace std;

void ModelicaLoweringPass::getDependentDialects(mlir::DialectRegistry &registry) const {
	registry.insert<linalg::LinalgDialect>();
	registry.insert<AffineDialect>();
	registry.insert<mlir::vector::VectorDialect>();
}

void ModelicaLoweringPass::runOnOperation()
{
	auto module = getOperation();
	ConversionTarget target(getContext());

	target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp>();
	target.addLegalDialect<StandardOpsDialect>();

	target.addLegalDialect<AffineDialect>();
	target.addLegalDialect<scf::SCFDialect>();
	target.addLegalDialect<linalg::LinalgDialect>();
	target.addLegalDialect<mlir::vector::VectorDialect>();

	// The Modelica dialect is defined as illegal, so that the conversion
	// will fail if any of its operations are not converted.
	target.addIllegalDialect<ModelicaDialect>();

	// Provide the set of patterns that will lower the Modelica operations
	mlir::OwningRewritePatternList patterns;
	populateModelicaConversionPatterns(patterns, &getContext());

	// With the target and rewrite patterns defined, we can now attempt the
	// conversion. The conversion will signal failure if any of our "illegal"
	// operations were not converted successfully.
	if (failed(applyFullConversion(module, target, move(patterns))))
		signalPassFailure();
}

void modelica::populateModelicaConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context)
{
	// Generic operations
	//patterns.insert<CastOpLowering>(context);
	//patterns.insert<CastCommonOpLowering>(context);
	patterns.insert<AssignmentOpLowering>(context);

	// Math operations
	//patterns.insert<AddOpLowering>(context);
	//patterns.insert<SubOpLowering>(context);
	//patterns.insert<MulOpLowering>(context);
	//patterns.insert<CrossProductOpLowering>(context);
	//patterns.insert<DivOpLowering>(context);

	// Logic operations
	//patterns.insert<NegateOpLowering>(context);
	//patterns.insert<EqOpLowering>(context);
	//patterns.insert<NotEqOpLowering>(context);
	//patterns.insert<GtOpLowering>(context);
	//patterns.insert<GteOpLowering>(context);
	//patterns.insert<LtOpLowering>(context);
	//patterns.insert<LteOpLowering>(context);

	// Control flow operations
	//patterns.insert<IfOpLowering>(context);
	//patterns.insert<ForOpLowering>(context);
	//patterns.insert<WhileOpLowering>(context);
	//patterns.insert<YieldOpLowering>(context);
}

std::unique_ptr<mlir::Pass> modelica::createModelicaLoweringPass()
{
	return std::make_unique<ModelicaLoweringPass>();
}
