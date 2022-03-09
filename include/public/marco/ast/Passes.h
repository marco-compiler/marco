#pragma once

// Just a convenience header file to include all the frontend passes and the
// pass manager to run them.

#include "marco/ast/passes/ConstantFoldingPass.h"
#include "marco/ast/passes/TypeCheckingPass.h"
#include "marco/ast/PassManager.h"
