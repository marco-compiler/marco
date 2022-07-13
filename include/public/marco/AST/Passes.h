#ifndef MARCO_AST_PASSES_H
#define MARCO_AST_PASSES_H

// Just a convenience header file to include all the frontend passes and the
// pass manager to run them.

#include "marco/AST/Passes/ConstantFoldingPass.h"
#include "marco/AST/Passes/SemanticAnalysisPass.h"
#include "marco/AST/Passes/StructuralConstantPropagationPass.h"
#include "marco/AST/Passes/TypeInferencePass.h"
#include "marco/AST/Passes/InliningPass.h"
#include "marco/AST/Passes/TypeCheckingPass.h"
#include "marco/AST/PassManager.h"

#endif // MARCO_AST_PASSES_H
