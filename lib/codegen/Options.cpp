#include "marco/codegen/Options.h"

namespace marco::codegen
{
  const CodegenOptions& CodegenOptions::getDefaultOptions() {
    static CodegenOptions options;
    return options;
  }
}