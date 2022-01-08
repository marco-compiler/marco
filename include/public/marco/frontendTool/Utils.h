//===----------------------------------------------------------------------===//
//
//  This header contains miscellaneous utilities for various frontend actions.
//  It is split from Frontend in order to minimise frontend's dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef MARCO_FRONTENDTOOL_UTILS_H
#define MARCO_FRONTENDTOOL_UTILS_H

namespace marco::frontend
{
  class CompilerInstance;

  class FrontendAction;

  /// Construct the FrontendAction of a compiler invocation based on the
  /// options specified for the compiler invocation.
  ///
  /// @return the created FrontendAction object
  std::unique_ptr<FrontendAction> createFrontendAction(CompilerInstance& ci);

  /// Execute the given actions described by the compiler invocation object
  /// in the given compiler instance.
  ///
  /// @return true on success; false otherwise
  bool executeCompilerInvocation(CompilerInstance* instance);
}

#endif // MARCO_FRONTENDTOOL_UTILS_H
