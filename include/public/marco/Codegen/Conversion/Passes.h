#ifndef MARCO_CODEGEN_CONVERSION_PASSES_H
#define MARCO_CODEGEN_CONVERSION_PASSES_H

#include "marco/Codegen/Conversion/BaseModelicaToArith/BaseModelicaToArith.h"
#include "marco/Codegen/Conversion/BaseModelicaToCF/BaseModelicaToCF.h"
#include "marco/Codegen/Conversion/BaseModelicaToFunc/BaseModelicaToFunc.h"
#include "marco/Codegen/Conversion/BaseModelicaToLLVM/BaseModelicaToLLVM.h"
#include "marco/Codegen/Conversion/BaseModelicaToLinalg/BaseModelicaToLinalg.h"
#include "marco/Codegen/Conversion/BaseModelicaToMLIRCore/BaseModelicaToMLIRCore.h"
#include "marco/Codegen/Conversion/BaseModelicaToMemRef/BaseModelicaToMemRef.h"
#include "marco/Codegen/Conversion/BaseModelicaToRuntime/BaseModelicaToRuntime.h"
#include "marco/Codegen/Conversion/BaseModelicaToRuntimeCall/BaseModelicaToRuntimeCall.h"
#include "marco/Codegen/Conversion/BaseModelicaToTensor/BaseModelicaToTensor.h"
#include "marco/Codegen/Conversion/IDAToFunc/IDAToFunc.h"
#include "marco/Codegen/Conversion/IDAToLLVM/IDAToLLVM.h"
#include "marco/Codegen/Conversion/KINSOLToFunc/KINSOLToFunc.h"
#include "marco/Codegen/Conversion/KINSOLToLLVM/KINSOLToLLVM.h"
#include "marco/Codegen/Conversion/RuntimeModelMetadataConversion/RuntimeModelMetadataConversion.h"
#include "marco/Codegen/Conversion/RuntimeToFunc/RuntimeToFunc.h"
#include "marco/Codegen/Conversion/RuntimeToLLVM/RuntimeToLLVM.h"
#include "marco/Codegen/Conversion/SUNDIALSToFunc/SUNDIALSToFunc.h"

namespace marco::codegen {
/// Generate the code for registering passes
#define GEN_PASS_REGISTRATION
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace marco::codegen

#endif // MARCO_CODEGEN_CONVERSION_PASSES_H
