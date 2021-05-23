#pragma once

#include <string>
#include <system_error>
#include <variant>

#include "llvm/Support/Error.h"
#include "marco/model/ModConst.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModLexerStateMachine.hpp"
#include "marco/utils/SourcePosition.h"

namespace marco
{
	enum class LowererErrorCode
	{
		success = 0,
		typeMisMatch,
		unkownVariable,
		globalVariableCreationFailure,
		functionAlreadyExists,
		typeConstantSizeMissMatch,
		unexpectedModToken,
		failedExplicitation,
		unsolvableAlgebraicLoop
	};
}

namespace std
{
	/**
	 * This class is required to specity that ParserErrorCode is a enum that is
	 * used to rappresent errors.
	 */
	template<>
	struct is_error_condition_enum<marco::LowererErrorCode>: public true_type
	{
	};
};	// namespace std

namespace marco
{
	/**
	 * A category is required to be compatible with std::error.
	 * All this standard and has to be done this way, look for a
	 * guide regarding how to make error category if you need to change this.
	 *
	 * When you add a new error kind you must add that choice to message in the
	 * cpp file.
	 */
	class LowererErrorCategory: public std::error_category
	{
		public:
		static LowererErrorCategory category;
		[[nodiscard]] std::error_condition default_error_condition(
				int ev) const noexcept override;

		[[nodiscard]] const char* name() const noexcept override
		{
			return "Lowerer Error";
		}

		[[nodiscard]] bool equivalent(
				const std::error_code& code, int condition) const noexcept override;

		[[nodiscard]] std::string message(int ev) const noexcept override;
	};

	std::error_condition make_error_condition(LowererErrorCode errc);

	class TypeMissMatch: public llvm::ErrorInfo<TypeMissMatch>
	{
		public:
		static char ID;
		TypeMissMatch(ModExp leftHand): leftHand(std::move(leftHand)) {}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Type Miss Match In:\n\t";
			leftHand.dump(OS);
			OS << '\n';
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::typeMisMatch),
					LowererErrorCategory::category);
		}

		[[nodiscard]] const ModExp& getExp() const { return leftHand; }

		private:
		ModExp leftHand;
	};

	class UnkownVariable: public llvm::ErrorInfo<UnkownVariable>
	{
		public:
		static char ID;
		UnkownVariable(std::string varName): varName(std::move(varName)) {}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Unkown Variable Named " << varName;
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::unkownVariable),
					LowererErrorCategory::category);
		}

		[[nodiscard]] llvm::StringRef getVarName() const { return varName; }

		private:
		std::string varName;
	};

	class GlobalVariableCreationFailure
			: public llvm::ErrorInfo<GlobalVariableCreationFailure>
	{
		public:
		static char ID;
		GlobalVariableCreationFailure(std::string varName)
				: varName(std::move(varName))
		{
		}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Could Not Create Variable: " << varName;
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::globalVariableCreationFailure),
					LowererErrorCategory::category);
		}

		[[nodiscard]] llvm::StringRef getVarName() const { return varName; }

		private:
		std::string varName;
	};

	class FunctionAlreadyExists: public llvm::ErrorInfo<FunctionAlreadyExists>
	{
		public:
		static char ID;
		FunctionAlreadyExists(std::string name): varName(std::move(name)) {}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Could Not Create Already Existing Function: " << varName;
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::functionAlreadyExists),
					LowererErrorCategory::category);
		}

		[[nodiscard]] llvm::StringRef getFunctionName() const { return varName; }

		private:
		std::string varName;
	};
	class TypeConstantSizeMissMatch
			: public llvm::ErrorInfo<TypeConstantSizeMissMatch>
	{
		public:
		static char ID;
		TypeConstantSizeMissMatch(ModConst cnst, ModType type)
				: constant(std::move(cnst)), type(std::move(type))
		{
		}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Size missmatch between type:\n\t";
			type.dump(OS);
			OS << "\nand\n\t";

			constant.dump(OS);
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::functionAlreadyExists),
					LowererErrorCategory::category);
		}

		[[nodiscard]] const ModConst& getConstant() const { return constant; }
		[[nodiscard]] const ModType& getType() const { return type; }

		private:
		ModConst constant;
		ModType type;
	};

	class UnexpectedModToken: public llvm::ErrorInfo<UnexpectedModToken>
	{
		public:
		static char ID;
		UnexpectedModToken(ModToken received, ModToken expected, SourcePosition pos)
				: expected(expected), received(received), pos(pos)
		{
		}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << '[' << pos << "] ";
			OS << "unexpected token: ";
			OS << tokenToString(received);
			OS << " expected " << tokenToString(expected);
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::unexpectedModToken),
					LowererErrorCategory::category);
		}

		private:
		ModToken expected;
		ModToken received;
		SourcePosition pos;
	};

	class FailedExplicitation: public llvm::ErrorInfo<FailedExplicitation>
	{
		public:
		static char ID;
		FailedExplicitation(ModExp toExplicitate, size_t argumentIndex)
				: toExplicitate(std::move(toExplicitate)), argumentIndex(argumentIndex)
		{
		}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Could not explicitate argument n: ";
			OS << argumentIndex;
			OS << " of expression: ";
			toExplicitate.dump(OS);
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::failedExplicitation),
					LowererErrorCategory::category);
		}

		private:
		ModExp toExplicitate;
		size_t argumentIndex;
	};

	/**
	 * Error caused by the presence of an Algebraic Loop that could not be solved
	 * during the SccCollapsing pass.
	 */
	class UnsolvableAlgebraicLoop: public llvm::ErrorInfo<UnsolvableAlgebraicLoop>
	{
		public:
		static char ID;
		UnsolvableAlgebraicLoop() {}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Some Algebraic Loops could not be solved by SccCollapsing.";
			OS << "A suitable solver must be used to solve such models";
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::unsolvableAlgebraicLoop),
					LowererErrorCategory::category);
		}
	};
}	 // namespace marco
