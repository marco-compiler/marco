#pragma once

#include <string>
#include <system_error>

#include "llvm/Support/Error.h"
#include "modelica/lowerer/SimExp.hpp"
#include "modelica/lowerer/Simulation.hpp"

namespace modelica
{
	enum class LowererErrorCode
	{
		success = 0,
		typeMisMatch,
		unkownVariable,
		globalVariableCreationFailure
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
		TypeMissMatch(SimExp leftHand, SimExp rightHand)
				: leftHand(std::move(leftHand)), rightHand(std::move(rightHand))
		{
		}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Type Miss Match Between:\n\t";
			leftHand.dump(OS);
			OS << "\n and \n\t";
			rightHand.dump(OS);
			OS << '\n';
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(LowererErrorCode::typeMisMatch),
					LowererErrorCategory::category);
		}

		[[nodiscard]] const SimExp& getLeftHand() const { return leftHand; }
		[[nodiscard]] const SimExp& getRightHand() const { return rightHand; }

		private:
		SimExp leftHand;
		SimExp rightHand;
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
}	// namespace modelica
