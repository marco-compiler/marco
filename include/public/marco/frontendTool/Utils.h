#ifndef MARCO_FRONTENDTOOL_UTILS_H
#define MARCO_FRONTENDTOOL_UTILS_H

namespace marco::frontend
{
  class CompilerInstance;

  class CompilerAction;

  /// Construct the FrontendAction of a compiler invocation based on the
  /// options specified for the compiler invocation.
  ///
  /// \return - The created FrontendAction object
  std::unique_ptr<CompilerAction> CreateFrontendAction(CompilerInstance& ci);

  /// ExecuteCompilerInvocation - Execute the given actions described by the
  /// compiler invocation object in the given compiler instance.
  ///
  /// \return - True on success.
  bool ExecuteCompilerInvocation(CompilerInstance* instance);
}

#endif // MARCO_FRONTENDTOOL_UTILS_H
