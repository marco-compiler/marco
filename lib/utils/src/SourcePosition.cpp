#include <modelica/utils/SourcePosition.h>

using namespace llvm;
using namespace modelica;
using namespace std;

SourcePosition::SourcePosition(string file, unsigned int line, unsigned int column)
		: file(std::make_shared<string>(file)),
			line(line),
			column(column)
{
}

SourcePosition SourcePosition::unknown()
{
	return SourcePosition("-", 0, 0);
}

namespace modelica
{
	raw_ostream& operator<<(raw_ostream& stream, const SourcePosition& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const SourcePosition& obj)
	{
		return *obj.file + " " + to_string(obj.line) + ":" + to_string(obj.column);
	}
}

SourceRange::SourceRange(SourcePosition begin, SourcePosition end)
		: begin(move(begin)),
			end(move(end))
{
}

const SourcePosition& SourceRange::getBegin() const
{
	return begin;
}

const SourcePosition& SourceRange::getEnd() const
{
	return end;
}

namespace modelica
{
	raw_ostream& operator<<(raw_ostream& stream, const SourceRange& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const SourceRange& obj)
	{
		const auto& begin = obj.getBegin();
		const auto& end = obj.getEnd();

		assert(*begin.file == *end.file);

		return *begin.file + " " +
					 to_string(begin.line) + ":" + to_string(begin.column) + "-" +
					 to_string(end.line) + ":" + to_string(end.column);
	}
}
