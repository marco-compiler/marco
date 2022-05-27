#include "marco/Utils/LogMessage.h"

using namespace marco;

AbstractMessage::AbstractMessage() = default;
AbstractMessage::AbstractMessage(const AbstractMessage& other) = default;
AbstractMessage::AbstractMessage(AbstractMessage&& other) = default;

AbstractMessage::~AbstractMessage() = default;

AbstractMessage& AbstractMessage::operator=(const AbstractMessage& other) = default;
AbstractMessage& AbstractMessage::operator=(AbstractMessage&& other) = default;

bool AbstractMessage::printBeforeMessage(llvm::raw_ostream& os) const
{
	return false;
}

bool AbstractMessage::printAfterLines(llvm::raw_ostream& os) const
{
	return false;
}

void AbstractMessage::print(llvm::raw_ostream& os) const
{
	if (printBeforeMessage(os))
		os << "\n";

	SourceRange location = getLocation();

	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	//os << *location.fileName << ":" << location.startLine << ":" << location.startColumn << ": ";

	getFormatter()(os);
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << "error: ";
	os.resetColor();

	printMessage(os);
	os << "\n";

	//location.printLines(os, getFormatter());

	if (printAfterLines(os))
		os << "\n";
}

std::function<void (llvm::raw_ostream &)> ErrorMessage::getFormatter() const
{
	return [](llvm::raw_ostream& stream) {
		stream.changeColor(llvm::raw_ostream::RED);
	};
}

std::function<void (llvm::raw_ostream &)> WarningMessage::getFormatter() const
{
	return [](llvm::raw_ostream& stream) {
		stream.changeColor(llvm::raw_ostream::YELLOW);
	};
}
