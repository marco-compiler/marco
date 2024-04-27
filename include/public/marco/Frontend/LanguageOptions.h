#ifndef MARCO_FRONTEND_LANGUAGEOPTIONS_H
#define MARCO_FRONTEND_LANGUAGEOPTIONS_H

#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace marco::frontend
{
  /// Language options for Modelica.
  /// The class extends the language options for C / C++ to enable the
  /// integration with clang's diagnostics infrastructure.
  class LanguageOptions
      : public llvm::RefCountedBase<LanguageOptions>,
        public clang::LangOptions
  {
  };
}

#endif // MARCO_FRONTEND_LANGUAGEOPTIONS_H
