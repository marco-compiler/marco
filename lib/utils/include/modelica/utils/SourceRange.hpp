#pragma once

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>

namespace modelica
{
	struct SourcePosition
	{
		SourcePosition(std::string file, unsigned int line, unsigned int column);

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
		SourceRange(SourcePosition b, SourcePosition e);

		[[nodiscard]] const SourcePosition& getBegin() const;
		[[nodiscard]] const SourcePosition& getEnd() const;

		private:
		SourcePosition begin;
		SourcePosition end;
	};

    llvm::raw_ostream& operator<<(
            llvm::raw_ostream& stream, const SourceRange& obj);

    std::string toString(const SourceRange& obj);
}	// namespace modelica
