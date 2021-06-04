#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <modelica/frontend/Errors.h>
#include <modelica/frontend/Pass.h>
#include <modelica/frontend/Symbol.hpp>

namespace modelica::frontend
{
	class Algorithm;
	class AssignmentStatement;
	class BreakStatement;
	class Call;
	class Class;
	class ClassContainer;
	class Constant;
	class PartialPartialDerFunction;
	class Equation;
	class Expression;
	class ForEquation;
	class ForStatement;
	class IfStatement;
	class Member;
	class Operation;
	class ReferenceAccess;
	class ReturnStatement;
	class StandardFunction;
	class Statement;
	class Tuple;
	class Type;
	class WhenStatement;
	class WhileStatement;

	namespace typecheck::detail
	{
		struct BuiltInFunction
		{
			/**
			 * Get the result type in case of non element-wise call.
			 * The arguments str needed because some functions (such
			 * as min / max / size) may vary their behaviour according to it.
			 * If the arguments count is invalid, then no type is returned.
			 *
			 * @param args	actual arguments
			 * @return result type
			 */
			[[nodiscard]] virtual llvm::Optional<Type> resultType(
					llvm::ArrayRef<std::unique_ptr<Expression>> args) const = 0;

			/**
			 * Whether the function can be used in an element-wise call.
			 *
			 * @return true if allowed; false otherwise
			 */
			[[nodiscard]] virtual bool canBeCalledElementWise() const = 0;

			/**
			 * Get the expected rank for each argument of the function.
			 * A rank of -1 means any rank is accepted (functions like ndims are
			 * made for the exact purpose to receive such arguments).
			 * The total arguments count is needed because some functions (such
			 * as min / max) may vary their behaviour according to it.
			 *
			 * @param argsCount  arguments count
			 * @param ranks 		 ranks container
			 */
			virtual void getArgsExpectedRanks(
					unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const = 0;
		};
	}

	class TypeChecker : public Pass
	{
		public:
		using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Symbol>;

		TypeChecker();

		llvm::Error run(llvm::ArrayRef<std::unique_ptr<Class>> classes) final;

		llvm::Error run(Algorithm& algorithm);

		template<typename T>
		[[nodiscard]] llvm::Error run(Class& cls);

		llvm::Error run(Equation& equation);

		template<typename T>
		[[nodiscard]] llvm::Error run(Expression& expression);

		llvm::Error run(ForEquation& forEquation);
		llvm::Error run(Induction& induction);
		llvm::Error run(Member& member);

		template<typename T>
		[[nodiscard]] llvm::Error run(Statement& statement);

		[[nodiscard]] llvm::Error checkGenericOperation(Expression& expression);

		[[nodiscard]] llvm::Error checkAddOp(Expression& expression);
		[[nodiscard]] llvm::Error checkDifferentOp(Expression& expression);
		[[nodiscard]] llvm::Error checkDivOp(Expression& expression);
		[[nodiscard]] llvm::Error checkEqualOp(Expression& expression);
		[[nodiscard]] llvm::Error checkGreaterOp(Expression& expression);
		[[nodiscard]] llvm::Error checkGreaterEqualOp(Expression& expression);
		[[nodiscard]] llvm::Error checkIfElseOp(Expression& expression);
		[[nodiscard]] llvm::Error checkLogicalAndOp(Expression& expression);
		[[nodiscard]] llvm::Error checkLogicalOrOp(Expression& expression);
		[[nodiscard]] llvm::Error checkLessOp(Expression& expression);
		[[nodiscard]] llvm::Error checkLessEqualOp(Expression& expression);
		[[nodiscard]] llvm::Error checkMemberLookupOp(Expression& expression);
		[[nodiscard]] llvm::Error checkMulOp(Expression& expression);
		[[nodiscard]] llvm::Error checkNegateOp(Expression& expression);
		[[nodiscard]] llvm::Error checkPowerOfOp(Expression& expression);
		[[nodiscard]] llvm::Error checkSubOp(Expression& expression);
		[[nodiscard]] llvm::Error checkSubscriptionOp(Expression& expression);

		private:
		SymbolTable symbolTable;
		llvm::StringMap<std::unique_ptr<typecheck::detail::BuiltInFunction>> builtInFunctions;
	};

	template<>
	llvm::Error TypeChecker::run<Class>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<PartialDerFunction>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<StandardFunction>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<Model>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<Package>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<Record>(Class& cls);

	template<>
	llvm::Error TypeChecker::run<Expression>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Array>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Call>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Constant>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Operation>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<ReferenceAccess>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<Tuple>(Expression& expression);

	template<>
	llvm::Error TypeChecker::run<AssignmentStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<BreakStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<ForStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<IfStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<ReturnStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<WhenStatement>(Statement& statement);

	template<>
	llvm::Error TypeChecker::run<WhileStatement>(Statement& statement);

	std::unique_ptr<Pass> createTypeCheckingPass();
}

namespace modelica::frontend::detail
{
	enum class TypeCheckingErrorCode
	{
		success = 0,
		assignment_to_input_member,
		bad_semantic,
		multiple_algorithms_function,
		not_found
	};
}

namespace std
{
	template<>
	struct is_error_condition_enum<modelica::frontend::detail::TypeCheckingErrorCode>
			: public std::true_type
	{
	};
}

namespace modelica::frontend
{
	namespace detail
	{
		class TypeCheckingErrorCategory: public std::error_category
		{
			public:
			static TypeCheckingErrorCategory category;

			[[nodiscard]] std::error_condition default_error_condition(int ev) const
			noexcept override;

			[[nodiscard]] const char* name() const noexcept override
			{
				return "Type checking error";
			}

			[[nodiscard]] bool equivalent(
					const std::error_code& code, int condition) const noexcept override;

			[[nodiscard]] std::string message(int ev) const noexcept override;
		};

		std::error_condition make_error_condition(TypeCheckingErrorCode errc);
	}

	class AssignmentToInputMember
			: public ErrorMessage,
				public llvm::ErrorInfo<AssignmentToInputMember>
	{
		public:
		static char ID;

		AssignmentToInputMember(SourceRange location, llvm::StringRef className);

		[[nodiscard]] SourceRange getLocation() const override;

		bool printBeforeMessage(llvm::raw_ostream& os) const override;
		void printMessage(llvm::raw_ostream& os) const override;

		void log(llvm::raw_ostream& os) const override;

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(detail::TypeCheckingErrorCode::assignment_to_input_member),
					detail::TypeCheckingErrorCategory::category);
		}

		private:
		SourceRange location;
		std::string className;
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
					static_cast<int>(detail::TypeCheckingErrorCode::bad_semantic),
					detail::TypeCheckingErrorCategory::category);
		}

		private:
		SourceRange location;
		std::string message;
	};

	class MultipleAlgorithmsFunction
			: public ErrorMessage,
				public llvm::ErrorInfo<MultipleAlgorithmsFunction>
	{
		public:
		static char ID;

		MultipleAlgorithmsFunction(SourceRange location, llvm::StringRef functionName);

		[[nodiscard]] SourceRange getLocation() const override;

		bool printBeforeMessage(llvm::raw_ostream& os) const override;
		void printMessage(llvm::raw_ostream& os) const override;

		void log(llvm::raw_ostream& os) const override;

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(detail::TypeCheckingErrorCode::multiple_algorithms_function),
					detail::TypeCheckingErrorCategory::category);
		}

		private:
		SourceRange location;
		std::string functionName;
	};

	class NotFound
			: public ErrorMessage,
				public llvm::ErrorInfo<NotFound>
	{
		public:
		static char ID;

		NotFound(SourceRange location, llvm::StringRef variableName);

		[[nodiscard]] SourceRange getLocation() const override;

		void printMessage(llvm::raw_ostream& os) const override;

		void log(llvm::raw_ostream& os) const override;

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(detail::TypeCheckingErrorCode::not_found),
					detail::TypeCheckingErrorCategory::category);
		}

		private:
		SourceRange location;
		std::string variableName;
	};
}
