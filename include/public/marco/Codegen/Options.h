#ifndef MARCO_CODEGEN_OPTIONS_H
#define MARCO_CODEGEN_OPTIONS_H

namespace marco::codegen
{
  struct CodegenOptions
  {
    /// Get a statically allocated copy of the default options.
    static const CodegenOptions& getDefaultOptions();
  };
}

#endif // MARCO_CODEGEN_OPTIONS_H