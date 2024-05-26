#ifndef MARCO_CODEGEN_TRANSFORMS_PASSES_H
#define MARCO_CODEGEN_TRANSFORMS_PASSES_H

namespace marco::codegen
{
  /// Generate the code for registering passes.
  #define GEN_PASS_REGISTRATION
  #include "marco/Codegen/Transforms/Passes.h.inc"
}

#endif // MARCO_CODEGEN_TRANSFORMS_PASSES_H
