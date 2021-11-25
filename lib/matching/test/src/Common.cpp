#include "Common.h"

#include <iostream>
#include <llvm/ADT/StringRef.h>

namespace llvm
{
    std::ostream& operator<<(
            std::ostream& stream, const llvm::StringRef& str)
    {
      return stream << str.str();
    }
}
