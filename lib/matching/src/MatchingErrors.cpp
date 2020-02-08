#include "modelica/matching/MatchingErrors.hpp"
using namespace modelica;

MatchingErrorCategory MatchingErrorCategory::category;
char FailedMatching::ID;
char EquationAndStateMissmatch::ID;

std::error_condition modelica::make_error_condition(MatchingErrorCode errc)
{
	return std::error_condition(
			static_cast<int>(errc), MatchingErrorCategory::category);
}

/**
 * This is required by std::error, just add a line every time you need to
 * create a new error type.
 */
[[nodiscard]] std::error_condition
MatchingErrorCategory::default_error_condition(int ev) const noexcept
{
	if (ev == 0)
		return std::error_condition(MatchingErrorCode::success);
	if (ev == 1)
		return std::error_condition(MatchingErrorCode::failedMatching);
	return std::error_condition(MatchingErrorCode::failedMatching);
}

[[nodiscard]] bool MatchingErrorCategory::equivalent(
		const std::error_code& code, int condition) const noexcept
{
	bool equal = *this == code.category();
	auto v = default_error_condition(code.value()).value();
	equal = equal && static_cast<int>(v) == condition;
	return equal;
}

/**
 * Decides the messaged based upon the type.
 * This is done for compatibilty with std::error, but when writing
 * tools code you should report error with ExitOnError and that will use
 * the string provided by the class extending ErrorInfo.
 */
[[nodiscard]] std::string MatchingErrorCategory::message(int ev) const noexcept
{
	switch (ev)
	{
		case 0:
			return "Success";
		case 1:
			return "Failed matching";
		case 2:
			return "Equation and State missmatch";
		default:
			return "Unkown Error";
	}
}
