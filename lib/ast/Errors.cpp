#include "marco/ast/Errors.h"

using namespace marco;
using namespace marco::ast;

namespace marco::ast::detail
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
