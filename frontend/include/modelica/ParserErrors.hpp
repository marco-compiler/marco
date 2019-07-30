#pragma once

#include <string>
#include <system_error>

#include "llvm/Support/Error.h"
#include "modelica/LexerStateMachine.hpp"

namespace modelica
{
	enum class ParserErrorCode
	{
		success = 0,
		not_implemented,
		unexpected_token,
		unkown_error,
		choise_not_found,
		incompatible_type,
		branches_types_do_not_match,
		empty_list
	};
}

namespace std
{
	template<>
	struct is_error_condition_enum<modelica::ParserErrorCode>: public true_type
	{
	};
};	// namespace std

namespace modelica
{
	class ParserErrorCategory: public std::error_category
	{
		public:
		static ParserErrorCategory category;
		[[nodiscard]] std::error_condition default_error_condition(int ev) const
				noexcept override;

		[[nodiscard]] const char* name() const noexcept override
		{
			return "Parser Error";
		}

		[[nodiscard]] bool equivalent(
				const std::error_code& code, int condition) const noexcept override;

		[[nodiscard]] std::string message(int ev) const noexcept override;
	};

	std::error_condition make_error_condition(ParserErrorCode errc);

	class UnexpectedToken: public llvm::ErrorInfo<UnexpectedToken>
	{
		public:
		static char ID;
		UnexpectedToken(Token token): token(token) {}

		[[nodiscard]] Token getToken() const { return token; }
		void log(llvm::raw_ostream& OS) const override { OS << "Unexpected Token"; }

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::unexpected_token),
					ParserErrorCategory::category);
		}

		private:
		Token token;
	};

	class NotImplemented: public llvm::ErrorInfo<NotImplemented>
	{
		public:
		static char ID;
		NotImplemented(std::string mes): msg(std::move(mes)) {}

		[[nodiscard]] const std::string& getMessage() const { return msg; }

		void log(llvm::raw_ostream& OS) const override { OS << msg; }

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::not_implemented),
					ParserErrorCategory::category);
		}

		private:
		std::string msg;
	};

	class ChoiseNotFound: public llvm::ErrorInfo<ChoiseNotFound>
	{
		public:
		static char ID;

		void log(llvm::raw_ostream& OS) const override { OS << "Choise Not Found"; }

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::choise_not_found),
					ParserErrorCategory::category);
		}
	};

	class EmptyList: public llvm::ErrorInfo<EmptyList>
	{
		public:
		static char ID;

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "The list was expected not empty";
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::empty_list),
					ParserErrorCategory::category);
		}
	};

	class BranchesTypeDoNotMatch: public llvm::ErrorInfo<BranchesTypeDoNotMatch>
	{
		public:
		static char ID;

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "if else branches do not have the same type";
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::branches_types_do_not_match),
					ParserErrorCategory::category);
		}
	};

	class IncompatibleType: public llvm::ErrorInfo<IncompatibleType>
	{
		public:
		static char ID;

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Expression has wrong type";
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::incompatible_type),
					ParserErrorCategory::category);
		}
	};
}	// namespace modelica
