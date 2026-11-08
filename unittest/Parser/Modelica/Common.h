#ifndef MARCO_UNITTEST_PARSER_MODELICA_COMMON_H
#define MARCO_UNITTEST_PARSER_MODELICA_COMMON_H

#include "marco/Parser/Modelica/Parser.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Support/MemoryBuffer.h"

namespace marco::parser::modelica::test {
/// Diagnostic consumer storing the messages of the reported errors.
class CapturingConsumer : public clang::DiagnosticConsumer {
public:
  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic &info) override {
    clang::DiagnosticConsumer::HandleDiagnostic(level, info);
    llvm::SmallString<128> message;
    info.FormatDiagnostic(message);
    messages.push_back(message.str().str());
  }

  std::vector<std::string> messages;
};

/// Parser operating on a string.
class StringParser {
public:
  explicit StringParser(llvm::StringRef source)
      : sourceFile(std::make_shared<SourceFile>("test.mo")),
        buffer(llvm::MemoryBuffer::getMemBuffer(source)),
        sourceManager(sourceFile->getFileName(), source),
        diagnostics(llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
                        new clang::DiagnosticIDs()),
                    diagOpts, &consumer, false) {
    sourceFile->setMemoryBuffer(buffer.get());
    parser = std::make_unique<Parser>(diagnostics, sourceManager.get(),
                                      sourceFile);
  }

  Parser *operator->() { return parser.get(); }

  const std::vector<std::string> &getErrors() const {
    return consumer.messages;
  }

private:
  std::shared_ptr<SourceFile> sourceFile;
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  clang::SourceManagerForFile sourceManager;
  clang::DiagnosticOptions diagOpts;
  CapturingConsumer consumer;
  clang::DiagnosticsEngine diagnostics;
  std::unique_ptr<Parser> parser;
};
} // namespace marco::parser::modelica::test

#endif // MARCO_UNITTEST_PARSER_MODELICA_COMMON_H
