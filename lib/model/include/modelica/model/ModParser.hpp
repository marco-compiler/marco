#pragma once
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModEqTemplate.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/ModExpPath.hpp"
#include "modelica/model/ModLexerStateMachine.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/utils/Interval.hpp"
#include "modelica/utils/Lexer.hpp"
#include "modelica/utils/SourcePosition.h"

namespace modelica
{
	class ModParser
	{
		public:
		ModParser(const std::string& source): lexer(source), current(lexer.scan())
		{
		}

		ModParser(const char* source): lexer(source), current(lexer.scan()) {}

		/**
		 * Return the current position in the source stream
		 */
		[[nodiscard]] SourcePosition getPosition() const
		{
			return SourcePosition("-", lexer.getCurrentLine(), lexer.getCurrentColumn());
		}

		[[nodiscard]] llvm::Expected<ModExp> expression();

		[[nodiscard]] llvm::Expected<ModConst> boolVector();
		[[nodiscard]] llvm::Expected<ModConst> intVector();
		[[nodiscard]] llvm::Expected<ModConst> floatVector();
		[[nodiscard]] llvm::Expected<std::string> reference();
		[[nodiscard]] llvm::Expected<ModCall> call();
		[[nodiscard]] llvm::Expected<ModType> type();
		[[nodiscard]] llvm::Expected<llvm::SmallVector<size_t, 3>> typeDimensions();
		[[nodiscard]] llvm::Expected<std::vector<ModExp>> args();
		[[nodiscard]] llvm::Expected<std::tuple<ModExpKind, std::vector<ModExp>>>
		operation();
		[[nodiscard]] llvm::Expected<std::tuple<std::string, ModExp>> statement();
		[[nodiscard]] llvm::Expected<llvm::StringMap<ModVariable>> initSection();
		[[nodiscard]] llvm::Expected<float> floatingPoint();
		[[nodiscard]] llvm::Expected<int> integer();

		using TemplatesMap = std::map<std::string, std::shared_ptr<ModEqTemplate>>;
		[[nodiscard]] llvm::Expected<llvm::SmallVector<ModEquation, 0>>
		updateSection(const TemplatesMap& templatesMap = {});

		[[nodiscard]] llvm::Expected<TemplatesMap> templates();

		[[nodiscard]] llvm::Expected<ModEqTemplate> singleTemplate();
		[[nodiscard]] llvm::Expected<Model> simulation();
		[[nodiscard]] llvm::Expected<Interval> singleInduction();
		[[nodiscard]] llvm::Expected<MultiDimInterval> inductions();
		[[nodiscard]] llvm::Expected<EquationPath> matchingPath();
		[[nodiscard]] llvm::Expected<ModEquation> updateStatement(
				const TemplatesMap& map);
		[[nodiscard]] llvm::Expected<ModEquation> matchedUpdateStatement(
				const TemplatesMap& map);

		[[nodiscard]] ModToken getCurrentModToken() const { return current; }

		private:
		/**
		 * regular accept, if the current token it then the next one will be read
		 * and true will be returned, else false.
		 */
		bool accept(ModToken t)
		{
			if (current == t)
			{
				next();
				return true;
			}
			return false;
		}

		/**
		 * fancy overloads if you know at compile time
		 * which token you want.
		 */
		template<ModToken t>
		bool accept()
		{
			if (current == t)
			{
				next();
				return true;
			}
			return false;
		}

		/**
		 * return a error if was token was not accepted.
		 * Notice that since errors are returned instead of
		 * being thrown this mean that there is no really difference
		 * between accept and expect. It is used here to signal that if
		 * an expect fail then the function will terminate immediatly,
		 * a accept is allowed to continue instead so that
		 * it is less confusing to people that are used to the accept expect
		 * notation.
		 *
		 * expect returns an Expected bool instead of a llvm::Error
		 * beacause to check for errors in a expected you do if (!expected)
		 * and in a llvm::Error you do if (error), this would be so confusing
		 * that this decision was better.
		 */
		llvm::Expected<bool> expect(ModToken t);

		/**
		 * reads the next token
		 */
		void next() { current = lexer.scan(); }

		Lexer<ModLexerStateMachine> lexer;
		ModToken current;
	};

}	 // namespace modelica
