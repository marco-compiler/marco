#include "modelica/ParserErrors.hpp"
using namespace modelica;

ParserErrorCategory ParserErrorCategory::category;
char UnexpectedToken::ID;
char NotImplemented::ID;
char ChoiseNotFound::ID;
char IncompatibleType::ID;
char BranchesTypeDoNotMatch::ID;
char EmptyList::ID;

std::error_condition modelica::make_error_condition(ParserErrorCode errc)
{
	return std::error_condition(
			static_cast<int>(errc), ParserErrorCategory::category);
}

[[nodiscard]] std::error_condition ParserErrorCategory::default_error_condition(
		int ev) const noexcept
{
	if (ev == 0)
		return std::error_condition(ParserErrorCode::success);
	if (ev == 1)
		return std::error_condition(ParserErrorCode::not_implemented);
	if (ev == 2)
		return std::error_condition(ParserErrorCode::unexpected_token);
	if (ev == 3)
		return std::error_condition(ParserErrorCode::choise_not_found);
	if (ev == 4)
		return std::error_condition(ParserErrorCode::branches_types_do_not_match);
	if (ev == 5)
		return std::error_condition(ParserErrorCode::incompatible_type);
	if (ev == 6)
		return std::error_condition(ParserErrorCode::empty_list);

	return std::error_condition(ParserErrorCode::unexpected_token);
}

[[nodiscard]] bool ParserErrorCategory::equivalent(
		const std::error_code& code, int condition) const noexcept
{
	bool equal = *this == code.category();
	auto v = default_error_condition(code.value()).value();
	equal = equal && static_cast<int>(v) == condition;
	return equal;
}

[[nodiscard]] std::string ParserErrorCategory::message(int ev) const noexcept
{
	switch (ev)
	{
		case (0):
			return "Success";
		case (1):
			return "Not Implemented";
		case (2):
			return "Unexpected Token";
		case (3):
			return "Choise Not Found";
		case (4):
			return "If else branches type do not match";
		case (5):
			return "Expression type is incompatible";
		case (6):
			return "List was empty when expected not";
		default:
			return "Unkown Error";
	}
}
