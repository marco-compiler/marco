#include "marco/Codegen/Transforms/SBGMatching.h"
#include "marco/Dialect/SBG/SBGDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_SBGMATCHINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::sbg;

namespace
{
  class SBGMatchingPass
      : public mlir::impl::SBGMatchingPassBase<
            SBGMatchingPass>
  {
    public:
    using SBGMatchingPassBase
        ::SBGMatchingPassBase;

    void runOnOperation() override
    {
    }
  };
}

namespace mlir::sbg
{
  std::unique_ptr<mlir::Pass> createSBGMatchingPass()
  {
    return std::make_unique<SBGMatchingPass>();
  }
}