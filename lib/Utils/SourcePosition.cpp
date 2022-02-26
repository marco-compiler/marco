#include "marco/Utils/SourcePosition.h"

using namespace marco;

SourcePosition::SourcePosition(std::string file, unsigned int line, unsigned int column)
		: file(std::make_shared<std::string>(file)),
			line(line),
			column(column)
{
}

SourcePosition SourcePosition::unknown()
{
	return SourcePosition("-", 0, 0);
}

namespace marco
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const SourcePosition& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const SourcePosition& obj)
	{
		return *obj.file + " " + std::to_string(obj.line) + ":" + std::to_string(obj.column);
	}
}

SourceRange::SourceRange(llvm::StringRef fileName,
												 const char* source,
												 size_t startLine, size_t startColumn,
												 size_t endLine, size_t endColumn)
		: SourceRange(false, fileName, source, startLine, startColumn, endLine, endColumn)
{
}

SourceRange::SourceRange(bool unknown,
												 llvm::StringRef fileName,
												 const char* source,
												 size_t startLine, size_t startColumn,
												 size_t endLine, size_t endColumn)
		: fileName(std::make_shared<std::string>(fileName.str())),
			source(source),
			startLine(std::move(startLine)),
			startColumn(std::move(startColumn)),
			endLine(std::move(endLine)),
			endColumn(std::move(endColumn)),
			isUnknown(unknown)
{
}

SourceRange SourceRange::unknown()
{
	return SourceRange(true, "-", nullptr, 0, 0, 0, 0);
}

SourcePosition SourceRange::getStartPosition() const
{
	return SourcePosition(*fileName, startLine, startColumn);
}

void SourceRange::extendEnd(SourceRange to)
{
	if (endLine == to.endLine)
	{
		endColumn = std::max(endColumn, to.endColumn);
	}
	else if (endLine < to.endLine)
	{
		endLine = to.endLine;
		endColumn = to.endColumn;
	}
}

template <class T>
size_t numDigits(T value)
{
	size_t digits = 0;

	while (value) {
		value /= 10;
		++digits;
	}

	return digits;
}

static void printLineNumber(llvm::raw_ostream& os, size_t lineNumber, size_t lineDigits)
{
	if (auto digits = numDigits(lineNumber); digits < lineDigits)
	{
		for (size_t i = 0; i < lineDigits - digits; ++i)
			os << " ";
	}

	os << "  " << lineNumber << " | ";
}

static void printLine(llvm::raw_ostream& os,
											size_t lineNumber,
											size_t lineDigits,
											std::string_view line,
											size_t length,
											size_t start,
											size_t end,
											std::function<void(llvm::raw_ostream&)> formatter)
{
	os.resetColor();

	// Print the actual line
	printLineNumber(os, lineNumber, lineDigits);

	for (size_t i = 0; i < start && i < length; ++i)
		os << line[i];

	formatter(os);

	for (size_t i = start; i < end && i < length; ++i)
		os << line[i];

	os.resetColor();

	for (size_t i = end; i < length; ++i)
		os << line[i];

	os << "\n";

	// Print the indicators
	for (size_t i = 0; i < lineDigits; ++i)
		os << " ";

	os << "   | ";

	for (size_t i = 0; i < start && i < length; ++i)
		os << " ";

	formatter(os);

	for (size_t i = start; i < end && i < length; ++i)
		os << (i == start ? "^" : "~");

	os.resetColor();

	os << "\n";
}

void SourceRange::printLines(llvm::raw_ostream& os, std::function<void(llvm::raw_ostream&)> formatter) const
{
	if (isUnknown)
		return;

  //os << "startLine: " << startLine << "\n";
  //os << "endLine: " << endLine << "\n";
  //os << "startColumn: " << startColumn << "\n";
  //os << "endColumn: " << endColumn << "\n";

	assert(startLine < endLine || (startLine == endLine && startColumn <= endColumn));

	std::string_view sourceView(source);
	size_t currentLine = 1;
	const char* data = source;

	// Go to the start line
	for (const char& c : sourceView)
	{
		data = &c;

		if (currentLine == startLine)
			break;

		if (c == '\n')
			++currentLine;
	}

	size_t lineDigits = numDigits(endLine);

	while (currentLine != endLine + 1)
	{
		std::string_view line(data);
		size_t length = 0;
		size_t maxLength = line.length();

		while (line[length] != '\n' && line[length] != '\0' && length < maxLength)
			++length;

		size_t start = 0;
		size_t end = length;

		if (currentLine == startLine)
			start = startColumn - 1;

		if (currentLine == endLine)
			end = endColumn;

		printLine(os, currentLine, lineDigits, line, length, start, end, formatter);
		++currentLine;
	}
}

namespace marco
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const SourceRange& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const SourceRange& obj)
	{
		return *obj.fileName + " " +
					 std::to_string(obj.startLine) + ":" + std::to_string(obj.startColumn) + "-" +
					 std::to_string(obj.endLine) + ":" + std::to_string(obj.endColumn);
	}
}
