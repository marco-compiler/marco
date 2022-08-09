#ifndef MARCO_CODEGEN_CONVERSION_PASSES_H
#define MARCO_CODEGEN_CONVERSION_PASSES_H

// Just a convenience header file to include the conversion passes

#include "marco/Codegen/Conversion/IDAToLLVM/IDAToLLVM.h"
#include "marco/Codegen/Conversion/ModelicaToArith/ModelicaToArith.h"
#include "marco/Codegen/Conversion/ModelicaToCF/ModelicaToCF.h"
#include "marco/Codegen/Conversion/ModelicaToFunc/ModelicaToFunc.h"
#include "marco/Codegen/Conversion/ModelicaToLLVM/ModelicaToLLVM.h"
#include "marco/Codegen/Conversion/ModelicaToMemRef/ModelicaToMemRef.h"

namespace marco::codegen
{
  //===----------------------------------------------------------------------===//
  // Registration
  //===----------------------------------------------------------------------===//

  /// Generate the code for registering passes
  #define GEN_PASS_REGISTRATION
  #include "marco/Codegen/Conversion/Passes.h.inc"
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_H
