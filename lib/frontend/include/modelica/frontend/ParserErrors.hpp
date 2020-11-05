#pragma once

#include <llvm/Support/Error.h>
#include <modelica/utils/SourceRange.hpp>
#include <string>
#include <system_error>
#include <utility>

#include "LexerStateMachine.hpp"

namespace modelica
{
	enum class ParserErrorCode
	{
		success = 0,
		not_implemented,
		unexpected_token,
		unexpected_identifier,
		unknown_error,
		choise_not_found,
		incompatible_type,
		branches_types_do_not_match,
		empty_list,
		bad_semantic
	};
}

namespace std
{
	/**
	 * This class is required to specify that ParserErrorCode is a enum that is
	 * used to represent errors.
	 */
	template<>
	struct is_error_condition_enum<modelica::ParserErrorCode>: public true_type
	{
	};
};	// namespace std

namespace modelica
{
	/**
	 * A category is required to be compatible with std::error.
	 * All this standard and has to be done this way, look for a
	 * guide regarding how to make error category if you need to change this.
	 *
	 * When you add a new error kind you must add that choise to message in the
	 * cpp file.
	 */
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
		UnexpectedToken(Token received, Token expected, SourcePosition pos)
				: token(received), expected(expected), pos(pos)
		{
		}

		[[nodiscard]] Token getToken() const { return token; }
		void log(llvm::raw_ostream& OS) const override
		{
			OS << "[" << pos.toString() << "] "
				 << "Unexpected Token: " << token << ", expected: " << expected;
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::unexpected_token),
					ParserErrorCategory::category);
		}

		private:
		Token token;
		Token expected;
		SourcePosition pos;
	};

	class UnexpectedIdentifier: public llvm::ErrorInfo<UnexpectedIdentifier>
	{
		public:
		static char ID;

		UnexpectedIdentifier(
				std::string identifier, std::string expected, SourcePosition pos)
				: identifier(std::move(identifier)),
					expected(std::move(expected)),
					pos(pos)
		{
		}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "[" << pos.toString() << "] "
				 << "Unexpected Identifier: " << identifier
				 << ", expected: " << expected;
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::unexpected_identifier),
					ParserErrorCategory::category);
		}

		private:
		std::string identifier;
		std::string expected;
		SourcePosition pos;
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

	/**
	 * A list was recived empty when was expected not to be
	 */
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

	/**
	 * Branches do not have the same types.
	 */
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

	/**
	 * Incompatible type is used to signal that subexpressions are
	 * not acceptable due to their types.
	 */
	class IncompatibleType: public llvm::ErrorInfo<IncompatibleType>
	{
		public:
		static char ID;
		IncompatibleType(std::string message): mess(std::move(message)) {}

		void log(llvm::raw_ostream& OS) const override { OS << mess; }

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::incompatible_type),
					ParserErrorCategory::category);
		}

		private:
		std::string mess;
	};

	/**
	 * BadSemantic is used to signal that the element declaration
	 * is incompatible with the rules defined by the language.
	 */
	class BadSemantic: public llvm::ErrorInfo<BadSemantic>
	{
		public:
		static char ID;
		BadSemantic(std::string message): mess(std::move(message)) {}

		void log(llvm::raw_ostream& OS) const override { OS << mess; }

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::bad_semantic),
					ParserErrorCategory::category);
		}

		private:
		std::string mess;
	};

}	 // namespace modelica
