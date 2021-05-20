#include <modelica/frontend/Errors.h>

using namespace modelica;
using namespace modelica::frontend;

AbstractMessage::AbstractMessage() = default;
AbstractMessage::AbstractMessage(const AbstractMessage& other) = default;
AbstractMessage::AbstractMessage(AbstractMessage&& other) = default;

AbstractMessage::~AbstractMessage() = default;

AbstractMessage& AbstractMessage::operator=(const AbstractMessage& other) = default;
AbstractMessage& AbstractMessage::operator=(AbstractMessage&& other) = default;

void AbstractMessage::print(llvm::raw_ostream& os) const
{
	SourceRange location = getLocation();

	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << *location.fileName << ":" << location.startLine << ":" << location.startColumn << ": ";

	getFormatter()(os);
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << "error: ";
	os.resetColor();

	printMessage(os);
	os << "\n";

	location.printLines(os, getFormatter());
}

std::function<void (llvm::raw_ostream &)> ErrorMessage::getFormatter() const
{
	return [](llvm::raw_ostream& stream) {
		stream.changeColor(llvm::raw_ostream::RED);
	};
};

std::function<void (llvm::raw_ostream &)> WarningMessage::getFormatter() const
{
	return [](llvm::raw_ostream& stream) {
		stream.changeColor(llvm::raw_ostream::YELLOW);
	};
};

namespace modelica::frontend::detail
{
	GenericErrorCategory GenericErrorCategory::category;

	std::error_condition GenericErrorCategory::default_error_condition(int ev) const noexcept
	{
		if (ev == 1)
			return std::error_condition(GenericErrorCode::not_implemented);

		if (ev == 2)
			return std::error_condition(GenericErrorCode::choice_not_found);

		if (ev == 3)
			return std::error_condition(GenericErrorCode::empty_list);

		if (ev == 4)
			return std::error_condition(GenericErrorCode::branches_types_do_not_match);

		if (ev == 5)
			return std::error_condition(GenericErrorCode::incompatible_type);

		return std::error_condition(GenericErrorCode::success);
	}

	bool GenericErrorCategory::equivalent(const std::error_code& code, int condition) const noexcept
	{
		bool equal = *this == code.category();
		auto v = default_error_condition(code.value()).value();
		equal = equal && static_cast<int>(v) == condition;
		return equal;
	}

	std::string GenericErrorCategory::message(int ev) const noexcept
	{
		switch (ev)
		{
			case (0):
				return "Success";

			case (1):
				return "Not implemented";

			case (2):
				return "Choice not found";

			case (3):
				return "Empty list";

			case (4):
				return "Branches types do not match";

			case (5):
				return "Incompatible type";

			default:
				return "Unknown Error";
		}
	}

	std::error_condition make_error_condition(GenericErrorCode errc)
	{
		return std::error_condition(
				static_cast<int>(errc), detail::GenericErrorCategory::category);
	}
}

char NotImplemented::ID;
char ChoiceNotFound::ID;
char EmptyList::ID;
char BranchesTypeDoNotMatch::ID;
char IncompatibleType::ID;
