#pragma once

#include <memory>

namespace modelica
{
	class SourcePosition
	{
		public:
		SourcePosition(unsigned l, unsigned c): line(l), column(c) {}
		[[nodiscard]] unsigned getLine() const { return line; }
		[[nodiscard]] unsigned getColumn() const { return column; }

		private:
		unsigned line;
		unsigned column;
	};
	class SourceRange
	{
		public:
		SourceRange(SourcePosition b, SourcePosition e)
				: begin(std::move(b)), end(std::move(e))
		{
		}
		SourceRange(
				unsigned beginLine,
				unsigned beginColumn,
				unsigned endLine,
				unsigned endColumn)
				: begin(beginLine, beginColumn), end(endLine, endColumn)
		{
		}

		[[nodiscard]] const SourcePosition& getBegin() const { return begin; }
		[[nodiscard]] const SourcePosition& getEnd() const { return end; }

		private:
		SourcePosition begin;
		SourcePosition end;
	};
}	// namespace modelica
