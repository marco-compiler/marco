#include "marco/Modeling/Dumpable.h"
#include "llvm/Support/raw_ostream.h"

namespace marco::modeling::internal
{
  Dumpable::~Dumpable() = default;

  void Dumpable::dump() const
  {
    dump(llvm::errs());
  }
}
