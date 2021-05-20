#pragma once

#include <llvm/Support/Error.h>
#include <llvm/Support/WithColor.h>
#include <modelica/utils/SourcePosition.h>
#include <string>
#include <system_error>
#include <utility>

#include "LexerStateMachine.h"

namespace modelica::frontend
{
	class AbstractMessage
	{
		public:
		AbstractMessage() = default;
		AbstractMessage(const AbstractMessage& other) = default;
		AbstractMessage(AbstractMessage&& other) = default;

		virtual ~AbstractMessage() = default;

		AbstractMessage& operator=(const AbstractMessage& other) = default;
		AbstractMessage& operator=(AbstractMessage&& other) = default;

		[[nodiscard]] virtual SourceRange getLocation() const = 0;
		virtual void printMessage(llvm::raw_ostream& os) const = 0;
		[[nodiscard]] virtual std::function<void(llvm::raw_ostream&)> getFormatter() const = 0;

		void print(llvm::raw_ostream& os) const
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
	};

	class ErrorMessage : public AbstractMessage
	{
		public:
		[[nodiscard]] std::function<void (llvm::raw_ostream &)> getFormatter() const override
		{
			return [](llvm::raw_ostream& stream) {
				stream.changeColor(llvm::raw_ostream::RED);
			};
		};
	};

	class WarningMessage : public AbstractMessage
	{
		public:
		[[nodiscard]] std::function<void (llvm::raw_ostream &)> getFormatter() const override
		{
			return [](llvm::raw_ostream& stream) {
				stream.changeColor(llvm::raw_ostream::YELLOW);
			};
		};
	};
}

namespace modelica::frontend
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
	struct is_error_condition_enum<modelica::frontend::ParserErrorCode>: public true_type
	{
	};
};	// namespace std

namespace modelica::frontend
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

	class UnexpectedToken
			: public ErrorMessage,
				public llvm::ErrorInfo<UnexpectedToken>
	{
		public:
		static char ID;

		UnexpectedToken(SourceRange location, Token token);

		[[nodiscard]] SourceRange getLocation() const override;

		void printMessage(llvm::raw_ostream& os) const override;

		void log(llvm::raw_ostream& os) const override;

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::unexpected_token),
					ParserErrorCategory::category);
		}

		private:
		SourceRange location;
		Token token;
	};

	class UnexpectedIdentifier
			: public ErrorMessage,
				public llvm::ErrorInfo<UnexpectedIdentifier>
	{
		public:
		static char ID;

		UnexpectedIdentifier(SourceRange location, llvm::StringRef identifier, llvm::StringRef expected);

		[[nodiscard]] SourceRange getLocation() const override;

		void printMessage(llvm::raw_ostream& os) const override;

		void log(llvm::raw_ostream& os) const override;

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::unexpected_identifier),
					ParserErrorCategory::category);
		}

		private:
		SourceRange location;
		std::string identifier;
		std::string expected;
	};

	class BadSemantic
			: public ErrorMessage,
				public llvm::ErrorInfo<BadSemantic>
	{
		public:
		static char ID;

		BadSemantic(SourceRange location, llvm::StringRef message);

		[[nodiscard]] SourceRange getLocation() const override;

		void printMessage(llvm::raw_ostream& os) const override;

		void log(llvm::raw_ostream& os) const override;

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(ParserErrorCode::bad_semantic),
					ParserErrorCategory::category);
		}

		private:
		SourceRange location;
		std::string message;
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
}
