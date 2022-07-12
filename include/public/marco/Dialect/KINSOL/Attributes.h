#ifndef MARCO_DIALECTS_KINSOL_ATTRIBUTES_H
#define MARCO_DIALECTS_KINSOL_ATTRIBUTES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/StorageUniquer.h"
#include <map>

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/KINSOL/KINSOLAttributes.h.inc"

#endif // MARCO_DIALECTS_KINSOL_ATTRIBUTES_H
