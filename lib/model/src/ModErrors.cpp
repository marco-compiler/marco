#include "marco/model/ModErrors.hpp"
using namespace marco;

LowererErrorCategory LowererErrorCategory::category;
char TypeMissMatch::ID;
char UnkownVariable::ID;
char FunctionAlreadyExists::ID;
char GlobalVariableCreationFailure::ID;
char UnexpectedModToken::ID;
char TypeConstantSizeMissMatch::ID;
char FailedExplicitation::ID;
char UnsolvableAlgebraicLoop::ID;

std::error_condition marco::make_error_condition(LowererErrorCode errc)
{
	return std::error_condition(
			static_cast<int>(errc), LowererErrorCategory::category);
}

/**
 * This is required by std::error, just add a line every time you need to
 * create a new error type.
 */
[[nodiscard]] std::error_condition
LowererErrorCategory::default_error_condition(int ev) const noexcept
{
	if (ev == 0)
		return std::error_condition(LowererErrorCode::success);
	if (ev == 1)
		return std::error_condition(LowererErrorCode::typeMisMatch);
	if (ev == 3)
		return std::error_condition(LowererErrorCode::unkownVariable);
	if (ev == 3)
		return std::error_condition(
				LowererErrorCode::globalVariableCreationFailure);
	if (ev == 4)
		return std::error_condition(LowererErrorCode::functionAlreadyExists);
	if (ev == 5)
		return std::error_condition(LowererErrorCode::typeConstantSizeMissMatch);
	if (ev == 6)
		return std::error_condition(LowererErrorCode::unexpectedModToken);
	if (ev == 7)
		return std::error_condition(LowererErrorCode::failedExplicitation);
	if (ev == 8)
		return std::error_condition(LowererErrorCode::unsolvableAlgebraicLoop);

	return std::error_condition(LowererErrorCode::unkownVariable);
}

[[nodiscard]] bool LowererErrorCategory::equivalent(
		const std::error_code& code, int condition) const noexcept
{
	bool equal = *this == code.category();
	auto v = default_error_condition(code.value()).value();
	equal = equal && static_cast<int>(v) == condition;
	return equal;
}

/**
 * Decides the messaged based upon the type.
 * This is done for compatibility with std::error, but when writing
 * tools code you should report error with ExitOnError and that will use
 * the string provided by the class extending ErrorInfo.
 */
[[nodiscard]] std::string LowererErrorCategory::message(int ev) const noexcept
{
	switch (ev)
	{
		case 0:
			return "Success";
		case 1:
			return "TypeMissMatch";
		case 2:
			return "Unkown Variable";
		case 3:
			return "Global Variable Creation Failure";
		case 4:
			return "Function Already Exists";
		case 5:
			return "Type Constant Size Missmatch";
		case 6:
			return "Unexpected sim token";
		case 7:
			return "Failed Explicitation";
		case 8:
			return "Unsolvable Algebraic Loop";

		default:
			return "Unkown Error";
	}
}
