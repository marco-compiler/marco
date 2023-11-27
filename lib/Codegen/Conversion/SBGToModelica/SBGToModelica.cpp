#include "marco/Codegen/Conversion/SBGToModelica/SBGToModelica.h"
#include "marco/Dialect/SBG/SBGDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_SBGTOMODELICACONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::sbg;

static void populateSBGToModelicaPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::MLIRContext* context)
{

}

namespace
{
  class SBGToModelicaConversionPass
      : public mlir::impl::SBGToModelicaConversionPassBase<
            SBGToModelicaConversionPass>
  {
    public:
    using SBGToModelicaConversionPassBase
        ::SBGToModelicaConversionPassBase;

    void runOnOperation() override
    {
    }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createSBGToModelicaConversionPass()
  {
    return std::make_unique<SBGToModelicaConversionPass>();
  }
}