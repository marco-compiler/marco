#pragma once

// Just a convenience header file to include all the frontend passes and the
// pass manager to run them.

#include "passes/BreakRemovingPass.h"
#include "passes/ConstantFoldingPass.h"
#include "passes/ReturnRemovingPass.h"
#include "passes/TypeCheckingPass.h"
#include "PassManager.h"
