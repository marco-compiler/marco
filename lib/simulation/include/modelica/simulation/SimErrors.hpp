#pragma once

#include <string>
#include <system_error>
#include <variant>

#include "llvm/Support/Error.h"
#include "modelica/simulation/SimExp.hpp"
#include "modelica/simulation/SimLexerStateMachine.hpp"
#include "modelica/utils/SourceRange.hpp"

namespace modelica
{
	enum class LowererErrorCode
	{
		success = 0,
		typeMisMatch,
		unkownVariable,
		globalVariableCreationFailure,
		functionAlreadyExists,
		typeConstantSizeMissMatch,
		unexpectedSimToken

	};
}

namespace std
{
	/**
	 * This class is required to specity that ParserErrorCode is a enum that is
	 * used to rappresent errors.
	 */
	template<>
	struct is_error_condition_enum<modelica::LowererErrorCode>: public true_type
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
	class LowererErrorCategory: public std::error_category
	{
		public:
		static LowererErrorCategory category;
		[[nodiscard]] std::error_condition default_error_condition(int ev) const
				noexcept override;

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
		TypeMissMatch(SimExp leftHand): leftHand(std::move(leftHand)) {}

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

		[[nodiscard]] const SimExp& getExp() const { return leftHand; }

		private:
		SimExp leftHand;
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
		using AnyConstant =
				std::variant<SimConst<int>, SimConst<float>, SimConst<bool>>;
		static char ID;
		TypeConstantSizeMissMatch(AnyConstant cnst, SimType type)
				: constant(std::move(cnst)), type(std::move(type))
		{
		}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Size missmatch between type:\n\t";
			type.dump(OS);
			OS << "\nand\n\t";

			if (std::holds_alternative<IntSimConst>(constant))
				dumpConstant(std::get<IntSimConst>(constant), OS);
			else if (std::holds_alternative<FloatSimConst>(constant))
				dumpConstant(std::get<FloatSimConst>(constant), OS);
			else if (std::holds_alternative<BoolSimConst>(constant))
				dumpConstant(std::get<BoolSimConst>(constant), OS);
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::functionAlreadyExists),
					LowererErrorCategory::category);
		}

		[[nodiscard]] const AnyConstant& getConstant() const { return constant; }
		[[nodiscard]] const SimType& getType() const { return type; }

		private:
		AnyConstant constant;
		SimType type;
	};

	class UnexpectedSimToken: public llvm::ErrorInfo<UnexpectedSimToken>
	{
		public:
		static char ID;
		UnexpectedSimToken(SimToken expected, SimToken received, SourcePosition pos)
				: expected(expected), received(received), pos(pos)
		{
		}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << '[' << pos.toString() << "] ";
			OS << "unexpected token: ";
			OS << tokenToString(received);
			OS << " expected " << tokenToString(expected);
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::unexpectedSimToken),
					LowererErrorCategory::category);
		}

		private:
		SimToken expected;
		SimToken received;
		SourcePosition pos;
	};
}	 // namespace modelica
