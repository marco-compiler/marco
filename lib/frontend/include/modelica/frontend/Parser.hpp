#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <modelica/frontend/LexerStateMachine.hpp>
#include <modelica/frontend/ParserErrors.hpp>
#include <modelica/utils/Lexer.hpp>
#include <optional>

#include "AST.h"

namespace modelica::frontend
{
	class Argument;
	class ClassModification;
	class ElementModification;
	class ElementRedeclaration;
	class ElementReplaceable;
	class Expression;
	class Statement;
	class AssignmentStatement;
	class IfStatement;
	class ForStatement;
	class Modification;
	class WhileStatement;
	class WhenStatement;

	/**
	 * The parser encapsulates the lexer but not the memory where the string we
	 * are reading is held. It exposes parts of the grammatical rules that are
	 * available in the grammar (can be found at page ~ 265 of the 3.4 doc).
	 */
	class Parser
	{
		public:
		Parser(std::string filename, const std::string& source);
		Parser(const std::string& source);
		Parser(const char* source);

		/**
		 * Return the current position in the source stream.
		 */
		[[nodiscard]] SourcePosition getPosition() const;

		[[nodiscard]] Token getCurrentToken() const;

		llvm::Expected<std::string> identifier();

		llvm::Expected<std::unique_ptr<Class>> classDefinition();

		llvm::Expected<std::unique_ptr<Expression>> primary();
		llvm::Expected<std::unique_ptr<Expression>> factor();
		llvm::Expected<llvm::Optional<std::unique_ptr<Expression>>> termModification();
		llvm::Expected<std::unique_ptr<Expression>> term();
		llvm::Expected<Type> typeSpecifier();

		llvm::Expected<bool> forEquationBody(
				llvm::SmallVectorImpl<std::unique_ptr<ForEquation>>& equations,
				int nestingLevel);

		llvm::Expected<std::unique_ptr<Expression>> arithmeticExpression();

		llvm::Expected<bool> arrayDimensions(
				llvm::SmallVectorImpl<std::unique_ptr<ArrayDimension>>& dimensions);

		llvm::Error elementList(
				llvm::SmallVectorImpl<std::unique_ptr<Member>>& members,
				bool publicSection = true);

		llvm::Expected<TypePrefix> typePrefix();
		llvm::Expected<std::unique_ptr<Member>> element(bool publicSection = true);
		std::optional<OperationKind> relationalOperator();

		llvm::Expected<std::unique_ptr<Expression>> logicalTerm();
		llvm::Expected<std::unique_ptr<Expression>> logicalExpression();

		llvm::Expected<std::unique_ptr<Equation>> equation();

		llvm::Expected<bool> forEquation(
				llvm::SmallVectorImpl<std::unique_ptr<ForEquation>>& equations,
				int nestingLevel);

		llvm::Expected<bool> equationSection(
				llvm::SmallVectorImpl<std::unique_ptr<Equation>>& equations,
				llvm::SmallVectorImpl<std::unique_ptr<ForEquation>>& forEquations);

		llvm::Expected<std::unique_ptr<Expression>> expression();
		llvm::Expected<std::unique_ptr<Expression>> logicalFactor();
		llvm::Expected<std::unique_ptr<Expression>> relation();
		llvm::Expected<std::unique_ptr<Expression>> componentReference();

		llvm::Expected<bool> functionCallArguments(llvm::SmallVectorImpl<std::unique_ptr<Expression>>& args);

		llvm::Expected<std::unique_ptr<Algorithm>> algorithmSection();
		llvm::Expected<std::unique_ptr<Statement>> statement();
		llvm::Expected<std::unique_ptr<AssignmentStatement>> assignmentStatement();
		llvm::Expected<std::unique_ptr<IfStatement>> ifStatement();
		llvm::Expected<std::unique_ptr<ForStatement>> forStatement();
		llvm::Expected<std::unique_ptr<WhileStatement>> whileStatement();
		llvm::Expected<std::unique_ptr<WhenStatement>> whenStatement();
		llvm::Expected<std::unique_ptr<BreakStatement>> breakStatement();
		llvm::Expected<std::unique_ptr<ReturnStatement>> returnStatement();

		llvm::Expected<std::unique_ptr<Tuple>> outputExpressionList();

		llvm::Expected<bool> arraySubscript(llvm::SmallVectorImpl<std::unique_ptr<Expression>>& subscripts);

		llvm::Expected<std::unique_ptr<Annotation>> annotation();
		llvm::Expected<std::unique_ptr<Modification>> modification();
		llvm::Expected<std::unique_ptr<ClassModification>> classModification();
		llvm::Expected<std::unique_ptr<Argument>> argument();
		llvm::Expected<std::unique_ptr<ElementModification>> elementModification(bool each, bool final);
		llvm::Expected<std::unique_ptr<ElementRedeclaration>> elementRedeclaration();
		llvm::Expected<std::unique_ptr<ElementReplaceable>> elementReplaceable(bool each, bool final);

		private:
		/**
		 * Read the next token.
		 */
		void next();

		/**
		 * Regular accept: if the current token is t then the next one will be read
		 * and true will be returned, else false.
		 */
		bool accept(Token t);
		/**
		 * fancy overloads if you know at compile time
		 * which token you want.
		 */
		template<Token t>
		bool accept()
		{
			if (current == t)
			{
				next();
				return true;
			}
			return false;
		}

		llvm::Expected<bool> expect(Token t);

		/**
		 * Unfortunately the grammar in the specification is written in such a
		 * way that there is a single point where you need a two lookahead
		 * to tell the difference between a component reference and a
		 * named argument. Instead of adding a real two lookahead or to completely
		 * change factor the grammar we can provide the ability to undo the last
		 * accept. If you need to implement that particular case, use this.
		 */
		void undoScan(Token t);
		const std::string filename;
		Lexer<ModelicaStateMachine> lexer;
		Token current;
		Token undo;
	};
}
