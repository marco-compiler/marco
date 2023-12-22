#include "marco/Dialect/KINSOL/Attributes.h"
#include "marco/Dialect/KINSOL/KINSOLDialect.h"
#include "marco/Dialect/KINSOL/Ops.h"
#include "marco/Dialect/KINSOL/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

using namespace ::mlir;
using namespace ::mlir::kinsol;

#include "marco/Dialect/KINSOL/KINSOLDialect.cpp.inc"

//===---------------------------------------------------------------------===//
// KINSOL dialect
//===---------------------------------------------------------------------===//

void KINSOLDialect::initialize() {
  addOperations<
      #define GET_OP_LIST
      #include "marco/Dialect/KINSOL/KINSOL.cpp.inc"
  >();

  addTypes<
      #define GET_TYPEDEF_LIST
      #include "marco/Dialect/KINSOL/KINSOLTypes.cpp.inc"
  >();

  addAttributes<
      #define GET_ATTRDEF_LIST
      #include "marco/Dialect/KINSOL/KINSOLAttributes.cpp.inc"
  >();
}

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/KINSOL/KINSOLTypes.cpp.inc"

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/KINSOL/KINSOLAttributes.cpp.inc"
