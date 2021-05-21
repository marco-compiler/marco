#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>

namespace modelica
{
	class SourcePosition
	{
		public:
		SourcePosition(std::string file, unsigned int line, unsigned int column);

		[[nodiscard]] static SourcePosition unknown();

		std::shared_ptr<std::string> file;
		unsigned int line;
		unsigned int column;
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const SourcePosition& obj);

	std::string toString(const SourcePosition& obj);

	class SourceRange
	{
		public:
		SourceRange(llvm::StringRef fileName,
								const char* source,
								size_t startLine, size_t startColumn,
								size_t endLine, size_t endColumn);

		[[nodiscard]] static SourceRange unknown();

		[[nodiscard]] SourcePosition getStartPosition() const;
		void extendEnd(SourceRange to);

		void printLines(llvm::raw_ostream& os, std::function<void(llvm::raw_ostream&)> formatter) const;

		std::shared_ptr<std::string> fileName;
		const char* source;

		size_t startLine;
		size_t startColumn;

		size_t endLine;
		size_t endColumn;

		private:
		SourceRange(bool unknown,
								llvm::StringRef fileName,
								const char* source,
								size_t startLine, size_t startColumn,
								size_t endLine, size_t endColumn);

		bool isUnknown;
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const SourceRange& obj);

	std::string toString(const SourceRange& obj);
}
