#ifndef MARCO_CODEN_CONVERSION_MODELICA_LOWERTOCFG_H
#define MARCO_CODEN_CONVERSION_MODELICA_LOWERTOCFG_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
  struct LowerToCFGOptions
  {
    unsigned int bitWidth = 64;
    bool outputArraysPromotion = true;
    bool inlining = true;

    static const LowerToCFGOptions& getDefaultOptions() {
      static LowerToCFGOptions options;
      return options;
    }
  };

	/// Convert the control flow operations of the Modelica and the SCF
	/// dialects.
	std::unique_ptr<mlir::Pass> createLowerToCFGPass(
      LowerToCFGOptions options = LowerToCFGOptions::getDefaultOptions());

	inline void registerLowerToCFGPass()
	{
		mlir::registerPass("convert-modelica-to-cfg", "Modelica: convert to CFG",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createLowerToCFGPass(LowerToCFGOptions::getDefaultOptions());
											 });
	}
}

#endif // MARCO_CODEN_CONVERSION_MODELICA_LOWERTOCFG_H
