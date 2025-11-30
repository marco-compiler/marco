#ifndef MARCO_FRONTEND_PASSES_H
#define MARCO_FRONTEND_PASSES_H

#include "marco/Frontend/Passes/AggressiveLICM.h"
#include "marco/Frontend/Passes/EquationIndexCheckInsertion.h"
#include "marco/Frontend/Passes/HeapFunctionsReplacement.h"
#include "marco/Frontend/Passes/Verifier.h"

namespace marco::frontend {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "marco/Frontend/Passes.h.inc"
} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_H
