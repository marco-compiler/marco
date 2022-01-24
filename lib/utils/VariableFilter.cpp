#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "marco/utils/Lexer.h"
#include "marco/utils/LogMessage.h"
#include "marco/utils/NumbersLexer.h"
#include "marco/utils/SourcePosition.h"
#include "marco/utils/VariableFilter.h"
#include <map>

using namespace marco;
using namespace marco::vf;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

enum class Token
{
	// Control tokens
	BeginOfFile,
	EndOfFile,
	Error,
	None,

	// Placeholders
	Integer,
	Ident,
	Regex,

	// Symbols
	LPar,
	RPar,
	LSquare,
	RSquare,
	Dollar,
	Comma,
	Semicolons,
	Colons,

	// Keywords
	DerKeyword
};

static std::string toString(Token token)
{
	switch (token)
	{
		case Token::BeginOfFile:
			return "Begin";
		case Token::EndOfFile:
			return "EOF";
		case Token::Error:
			return "Error";
		case Token::None:
			return "None";
		case Token::Integer:
			return "Integer";
		case Token::Ident:
			return "Identifier";
		case Token::Regex:
			return "Regex";
		case Token::LPar:
			return "(";
		case Token::RPar:
			return ")";
		case Token::LSquare:
			return "[";
		case Token::RSquare:
			return "]";
		case Token::Comma:
			return ",";
		case Token::Semicolons:
			return ";";
		case Token::Colons:
			return ":";
		case Token::Dollar:
			return "$";
		case Token::DerKeyword:
			return "der";
	}

	return "[Unexpected]";
}

static llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Token& obj)
{
	return stream << toString(obj);
}

static bool isDigit(char c)
{
	return ('0' <= c && c <= '9');
}

static bool isNonDigit(char c)
{
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_';
}

/**
 * State machine is the state machine of the variable filter grammar.
 * It implements the interface required by lexer.
 */
class LexerStateMachine
{
	public:
	using Token = ::Token;

	/**
	 * The possible states of the machine.
	 */
	enum class State
	{
		Normal,
		ParsingId,
		ParsingNum,
		ParsingRegex,
		IgnoreNextChar,
		End
	};

	LexerStateMachine(char first)
			: state(State::Normal),
				current('\0'),
				next(first),
				currentToken(Token::BeginOfFile),
				lastIdentifier(""),
				lastRegex(""),
				currentLine(1),
				currentColumn(0),
				startLine(1),
				startColumn(0),
				endLine(1),
				endColumn(0)
	{
		symbols['('] = Token::LPar;
		symbols[')'] = Token::RPar;
		symbols['['] = Token::LSquare;
		symbols[']'] = Token::RSquare;
		symbols[','] = Token::Comma;
		symbols[';'] = Token::Semicolons;
		symbols[':'] = Token::Colons;
		symbols['$'] = Token::Dollar;
		keywordMap["der"] = Token::DerKeyword;
	}

	/**
	 * Returns the last seen token, or 'Begin' if none was seen.
	 */
	[[nodiscard]] Token getCurrent() const
	{
		return currentToken;
	}

	[[nodiscard]] size_t getCurrentLine() const
	{
		return currentLine;
	}

	[[nodiscard]] size_t getCurrentColumn() const
	{
		return currentColumn;
	}

	[[nodiscard]] size_t getTokenStartLine() const
	{
		return startLine;
	}

	[[nodiscard]] size_t getTokenStartColumn() const
	{
		return startColumn;
	}

	[[nodiscard]] size_t getTokenEndLine() const
	{
		return endLine;
	}

	[[nodiscard]] size_t getTokenEndColumn() const
	{
		return endColumn;
	}

	/**
	 * Returns the last seen identifier, or the one being built if the machine
	 * is in the process of recognizing one.
	 */
	[[nodiscard]] const std::string& getLastIdentifier() const
	{
		return lastIdentifier;
	}

	/**
	 * Returns the last seen string, or the one being built if the machine is in
	 * the process of recognizing one.
	 */
	[[nodiscard]] const std::string& getLastRegex() const
	{
		return lastRegex;
	}

	/**
	 * Returns the last seen integer, or the one being built if the machine is
	 * in the process of recognizing one.
	 *
	 * Notice that as soon as a new number is found this value is overridden,
	 * even if it was a float and not a int.
	 */
	[[nodiscard]] long getLastInt() const
	{
		return lastNum.get();
	}

	/**
	 * Returns the string associated to the last Error token found.
	 */
	[[nodiscard]] const std::string& getLastError() const
	{
		return error;
	}

	protected:
	/**
	 * Feeds a character to the state machine, returns 'None' if
	 * the current token has not eaten all the possible character
	 * Returns 'Error' if the input was malformed.
	 * Returns 'End' if '\0' was found.
	 */
	Token step(char c);

	private:
	/**
	 * Updates column and line number, as well as current and next char.
	 */
	void advance(char c)
	{
		current = next;
		next = c;
		currentColumn++;

		if (current == '\n')
		{
			currentColumn = 0;
			currentLine++;
		}
	}

	[[nodiscard]] Token stringToToken(llvm::StringRef str) const
	{
		if (auto iter = keywordMap.find(str); iter != keywordMap.end())
			return iter->getValue();

		return Token::Ident;
	}

	[[nodiscard]] Token charToToken(char c) const
	{
		if (auto iter = symbols.find(c); iter != symbols.end())
			return iter->second;

		return Token::Error;
	}

	template<State s>
	Token scan() {
		return Token::None;
	}

	/**
	 * Try to scan the next symbol by taking into account both the current and the
	 * next characters. This avoids the need to define custom states to recognize
	 * simple symbols such as '==' or ':='
	 */
	Token tryScanSymbol()
	{
		state = State::Normal;
		Token token = charToToken(current);

		if (token == Token::Error)
		{
			error = "Unexpected character ";
			error.push_back(current);
		}

		return token;
	}

	void setTokenStartPosition()
	{
		startLine = currentLine;
		startColumn = currentColumn;
	}

	void setTokenEndPosition()
	{
		endLine = currentLine;
		endColumn = currentColumn;
	}

	State state;
	char current;
	char next;
	Token currentToken;
	std::string lastIdentifier;
	IntegerLexer<10> lastNum;
	std::string lastRegex;

	size_t currentLine;
	size_t currentColumn;

	size_t startLine;
	size_t startColumn;

	size_t endLine;
	size_t endColumn;

	std::string error;
	llvm::StringMap<Token> keywordMap;
	std::map<char, Token> symbols;
};

template<>
Token LexerStateMachine::scan<LexerStateMachine::State::ParsingId>()
{
	lastIdentifier.push_back(current);

	if (!isDigit(next) && !isNonDigit(next) && next != '.')
	{
		state = State::Normal;
		setTokenEndPosition();
		return stringToToken(lastIdentifier);
	}

	return Token::None;
}

template<>
Token LexerStateMachine::scan<LexerStateMachine::State::ParsingNum>()
{
	if (isDigit(current))
		lastNum += (current - '0');

	if (!isDigit(next))
	{
		state = State::Normal;
		setTokenEndPosition();
		return Token::Integer;
	}

	return Token::None;
}

template<>
Token LexerStateMachine::scan<LexerStateMachine::State::ParsingRegex>()
{
	if (current == '/')
	{
		state = State::Normal;
		setTokenEndPosition();
		return Token::Regex;
	}

	if (current == '\0')
	{
		state = State::End;
		error = "Reached end of string while parsing a regex";
		return Token::Error;
	}

	lastRegex.push_back(current);
	return Token::None;
}

template<>
Token LexerStateMachine::scan<LexerStateMachine::State::Normal>()
{
	if (std::isspace(current) != 0)
		return Token::None;

	setTokenStartPosition();

	if (isNonDigit(current))
	{
		state = State::ParsingId;
		lastIdentifier = "";

		return scan<State::ParsingId>();
	}

	if (isDigit(current))
	{
		state = State::ParsingNum;
		lastNum = IntegerLexer<10>();
		return scan<State::ParsingNum>();
	}

	if (current == '/')
	{
		state = State::ParsingRegex;
		lastRegex = "";
		return Token::None;
	}

	if (current == '\0')
	{
		state = State::End;
		return Token::EndOfFile;
	}

	return tryScanSymbol();
}

Token LexerStateMachine::step(char c)
{
	advance(c);

	switch (state)
	{
		case (State::Normal):
			return scan<State::Normal>();

		case (State::ParsingNum):
			return scan<State::ParsingNum>();

		case (State::ParsingRegex):
			return scan<State::ParsingRegex>();

		case (State::ParsingId):
			return scan<State::ParsingId>();

		case (State::End):
			return Token::EndOfFile;

		case (State::IgnoreNextChar):
			state = State::Normal;
			return Token::None;
	}

	error = "Unhandled Lexer State";
	return Token::Error;
}

//===----------------------------------------------------------------------===//
// AST
//===----------------------------------------------------------------------===//

class ASTNode
{
	public:
	virtual ~ASTNode() = default;
};

class VariableExpression : public ASTNode
{
	public:
	VariableExpression(llvm::StringRef identifier)
			: identifier(identifier.str())
	{
	}

	llvm::StringRef getIdentifier() const
	{
		return identifier;
	}

	private:
	std::string identifier;
};

class ArrayExpression : public ASTNode
{
	public:
	ArrayExpression(VariableExpression variable, llvm::ArrayRef<ArrayRange> ranges)
			: variable(std::move(variable)),
				ranges(ranges.begin(), ranges.end())
	{
	}

	VariableExpression getVariable() const
	{
		return variable;
	}

	llvm::ArrayRef<ArrayRange> getRanges() const
	{
		return ranges;
	}

	private:
	VariableExpression variable;
	llvm::SmallVector<ArrayRange> ranges;
};

class DerivativeExpression : public ASTNode
{
	public:
	DerivativeExpression(VariableExpression derivedVariable)
			: derivedVariable(std::move(derivedVariable))
	{
	}

	VariableExpression getDerivedVariable() const
	{
		return derivedVariable;
	}

	private:
	VariableExpression derivedVariable;
};

class RegexExpression : public ASTNode
{
	public:
	RegexExpression(llvm::StringRef regex)
			: regex(regex.str())
	{
	}

	llvm::StringRef getRegex() const
	{
		return regex;
	}

	private:
	std::string regex;
};

//===----------------------------------------------------------------------===//
// Errors
//===----------------------------------------------------------------------===//

namespace marco::vf::detail
{
	enum class ParsingErrorCode
	{
		success = 0,
		unexpected_token,
		empty_regex,
		invalid_regex
	};
}

namespace std
{
	template<>
	struct is_error_condition_enum<marco::vf::detail::ParsingErrorCode>
			: public std::true_type
	{
	};
}

namespace marco::vf::detail
{
	class ParsingErrorCategory: public std::error_category
	{
		public:
		static ParsingErrorCategory category;

		[[nodiscard]] std::error_condition default_error_condition(int ev) const noexcept override
		{
			if (ev == 1)
				return std::error_condition(ParsingErrorCode::unexpected_token);

			if (ev == 2)
				return std::error_condition(ParsingErrorCode::empty_regex);

			if (ev == 3)
				return std::error_condition(ParsingErrorCode::invalid_regex);

			return std::error_condition(ParsingErrorCode::success);
		}

		[[nodiscard]] const char* name() const noexcept override
		{
			return "Parsing error";
		}

		[[nodiscard]] bool equivalent(const std::error_code& code, int condition) const noexcept override
		{
			bool equal = *this == code.category();
			auto v = default_error_condition(code.value()).value();
			equal = equal && static_cast<int>(v) == condition;
			return equal;
		}

		[[nodiscard]] std::string message(int ev) const noexcept override
		{
			switch (ev)
			{
				case (0):
					return "Success";

				case (1):
					return "Unexpected Token";

				case (2):
					return "Empty regex";

				case (3):
					return "Invalid regex";

				default:
					return "Unknown Error";
			}
		}
	};

	ParsingErrorCategory ParsingErrorCategory::category;

	std::error_condition make_error_condition(ParsingErrorCode errc)
	{
		return std::error_condition(
				static_cast<int>(errc), detail::ParsingErrorCategory::category);
	}
}

class UnexpectedToken
		: public ErrorMessage,
			public llvm::ErrorInfo<UnexpectedToken>
{
	public:
	static char ID;

	UnexpectedToken(SourceRange location, Token token)
			: location(std::move(location)),
				token(token)
	{
	}

	[[nodiscard]] SourceRange getLocation() const override
	{
		return location;
	}

	void printMessage(llvm::raw_ostream& os) const override
	{
		os << "unexpected token [";
		os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
		os << token;
		os << "]";
	}

	void log(llvm::raw_ostream& os) const override
	{
		print(os);
	}

	[[nodiscard]] std::error_code convertToErrorCode() const override
	{
		return std::error_code(
				static_cast<int>(detail::ParsingErrorCode::unexpected_token),
				detail::ParsingErrorCategory::category);
	}

	private:
	SourceRange location;
	Token token;
};

class EmptyRegex
		: public ErrorMessage,
			public llvm::ErrorInfo<EmptyRegex>
{
	public:
	static char ID;

	EmptyRegex(SourceRange location)
			: location(std::move(location))
	{
	}

	[[nodiscard]] SourceRange getLocation() const override
	{
		return location;
	}

	void printMessage(llvm::raw_ostream& os) const override
	{
		os << "empty regex";
	}

	void log(llvm::raw_ostream& os) const override
	{
		print(os);
	}

	[[nodiscard]] std::error_code convertToErrorCode() const override
	{
		return std::error_code(
				static_cast<int>(detail::ParsingErrorCode::empty_regex),
				detail::ParsingErrorCategory::category);
	}

	private:
	SourceRange location;
};

class InvalidRegex
		: public ErrorMessage,
			public llvm::ErrorInfo<InvalidRegex>
{
	public:
	static char ID;

	InvalidRegex(SourceRange location)
			: location(std::move(location))
	{
	}

	[[nodiscard]] SourceRange getLocation() const override
	{
		return location;
	}

	void printMessage(llvm::raw_ostream& os) const override
	{
		os << "invalid regex";
	}

	void log(llvm::raw_ostream& os) const override
	{
		print(os);
	}

	[[nodiscard]] std::error_code convertToErrorCode() const override
	{
		return std::error_code(
				static_cast<int>(detail::ParsingErrorCode::invalid_regex),
				detail::ParsingErrorCategory::category);
	}

	private:
	SourceRange location;
};

char UnexpectedToken::ID;
char EmptyRegex::ID;
char InvalidRegex::ID;

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

#define EXPECT(Token)							\
	if (auto e = expect(Token); !e)	\
	return e.takeError()

#define TRY(outVar, expression)		\
	auto outVar = expression;				\
	if (!outVar)										\
	return outVar.takeError()

class Parser
{
	public:
	Parser(VariableFilter& vf, llvm::StringRef source)
			: vf(&vf),
				lexer(source.data()),
				current(lexer.scan()),
				tokenRange("-", source.data(), 1, 1, 1, 1)
	{
		updateTokenSourceRange();
	}

	llvm::Error run()
	{
		// In case of empty string
		if (current == Token::EndOfFile)
			return llvm::Error::success();

		// Consume consecutive semicolons
		while(accept<Token::Semicolons>());

		// Check if we reached the end of the string
		if (current == Token::EndOfFile)
			return llvm::Error::success();

		if (auto error = token(); error)
			return error;

		while (accept<Token::Semicolons>())
    {
			// For strings ending with a semicolon
			if (current == Token::EndOfFile)
				return llvm::Error::success();

			if (current != Token::Semicolons)
				if (auto error = token(); error)
					return error;
		}

		return llvm::Error::success();
	}

	llvm::Error token()
	{
		if (current == Token::DerKeyword)
		{
			TRY(derNode, der());
			Tracker tracker(derNode->getDerivedVariable().getIdentifier());
			vf->addDerivative(tracker);
			return llvm::Error::success();
		}
		else if (current == Token::Regex)
		{
			TRY(regexNode, regex());
			vf->addRegexString(regexNode->getRegex());
			return llvm::Error::success();
		}

		TRY(variableNode, identifier());

		if (current == Token::LSquare)
		{
			TRY(arrayNode, array(*variableNode));
			Tracker tracker(arrayNode->getVariable().getIdentifier(), arrayNode->getRanges());
			vf->addVariable(tracker);
			return llvm::Error::success();
		}

		vf->addVariable(Tracker(variableNode->getIdentifier()));
		return llvm::Error::success();
	}

	llvm::Expected<DerivativeExpression> der()
	{
		EXPECT(Token::DerKeyword);
		EXPECT(Token::LPar);
		VariableExpression variable(lexer.getLastIdentifier());
		EXPECT(Token::Ident);
		EXPECT(Token::RPar);
		return DerivativeExpression(variable);
	}

	llvm::Expected<RegexExpression> regex()
	{
		RegexExpression node(lexer.getLastRegex());
		EXPECT(Token::Regex);

		if (node.getRegex().empty())
			return llvm::make_error<EmptyRegex>(tokenRange);

		llvm::Regex regexObj(node.getRegex());

		if (!regexObj.isValid())
			return llvm::make_error<InvalidRegex>(tokenRange);

		return node;
	}

	llvm::Expected<VariableExpression> identifier()
	{
		VariableExpression node(lexer.getLastIdentifier());
		EXPECT(Token::Ident);
		return node;
	}

	llvm::Expected<ArrayExpression> array(VariableExpression variable)
	{
		EXPECT(Token::LSquare);

		llvm::SmallVector<ArrayRange, 3> ranges;
		TRY(range, arrayRange());
		ranges.push_back(*range);

		while (accept<Token::Comma>())
		{
			TRY(anotherRange, arrayRange());
			ranges.push_back(*anotherRange);
		}

		EXPECT(Token::RSquare);
		return ArrayExpression(variable, ranges);
	}

	llvm::Expected<ArrayRange> arrayRange()
	{
		auto getIndex = [&]() -> llvm::Expected<int> {
			if (accept<Token::Dollar>())
				return -1;

			int index = lexer.getLastInt();
			EXPECT(Token::Integer);
			return index;
		};

		TRY(lowerBound, getIndex());
		EXPECT(Token::Colons);
		TRY(upperBound, getIndex());

		return ArrayRange(*lowerBound, *upperBound);
	}

	private:
	/**
	 * Read the next token.
	 */
	void next()
	{
		current = lexer.scan();
		updateTokenSourceRange();
	}

	/**
	 * Regular accept: if the current token is t then the next one will be read
	 * and true will be returned, else false.
	 */
	bool accept(Token t)
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

	llvm::Expected<bool> expect(Token t)
	{
		if (accept(t))
			return true;

		return llvm::make_error<UnexpectedToken>(tokenRange, current);
	}

	void updateTokenSourceRange()
	{
		tokenRange.startLine = lexer.getTokenStartLine();
		tokenRange.startColumn = lexer.getTokenStartColumn();
		tokenRange.endLine = lexer.getTokenEndLine();
		tokenRange.endColumn = lexer.getTokenEndColumn();
	}

	VariableFilter* vf;
	Lexer<LexerStateMachine> lexer;
	Token current;
	SourceRange tokenRange;
};

//===----------------------------------------------------------------------===//
// VariableFilter
//===----------------------------------------------------------------------===//

ArrayRange::ArrayRange(long lowerBound, long upperBound)
		: lowerBound(lowerBound), upperBound(upperBound)
{
}

bool ArrayRange::hasLowerBound() const
{
	return lowerBound != unbounded;
}

long ArrayRange::getLowerBound() const
{
	assert(hasLowerBound());
	return lowerBound;
}

bool ArrayRange::hasUpperBound() const
{
	return upperBound != unbounded;
}

long ArrayRange::getUpperBound() const
{
	assert(hasUpperBound());
	return upperBound;
}

Tracker::Tracker()
{
}

Tracker::Tracker(llvm::StringRef name)
		: Tracker(name, llvm::None)
{
}

Tracker::Tracker(llvm::StringRef name, llvm::ArrayRef<ArrayRange> ranges)
		: name(name.str()),
			ranges(ranges.begin(), ranges.end())
{
}

void Tracker::setRanges(llvm::ArrayRef<ArrayRange> newRanges)
{
	this->ranges.clear();

	for (const auto& range : newRanges)
		this->ranges.push_back(range);
}

llvm::StringRef Tracker::getName() const
{
	return name;
}

llvm::ArrayRef<ArrayRange> Tracker::getRanges() const
{
	return ranges;
}

ArrayRange Tracker::getRangeOfDimension(unsigned int dimensionIndex) const
{
	assert(dimensionIndex < ranges.size());
	return *(ranges.begin() + dimensionIndex);
}

Filter::Filter(bool visibility, llvm::ArrayRef<ArrayRange> ranges)
		: visibility(visibility), ranges(ranges.begin(), ranges.end())
{
}

bool Filter::isVisible() const
{
	return visibility;
}

llvm::ArrayRef<ArrayRange> Filter::getRanges() const
{
	return ranges;
}

Filter Filter::visibleScalar()
{
	return Filter(true, llvm::None);
}

Filter Filter::visibleArray(llvm::ArrayRef<long> shape)
{
	llvm::SmallVector<ArrayRange, 3> ranges;

	for (const auto& dimension : shape)
	{
		long start = 0;
		long end = dimension == -1 ? -1 : dimension - 1;
		ranges.emplace_back(start, end);
	}

	return Filter(true, ranges);
}

void VariableFilter::dump() const
{
	dump(llvm::outs());
}

void VariableFilter::dump(llvm::raw_ostream& os) const
{
	// TODO: improve output

	/*
	for (int s = 0; s < 12; ++s)
		os << "#";

	os << "\n *** TRACKED VARIABLES *** : " << "\n";

	for (const auto& tracker : _variables)
		tracker.getValue().dump(os);

	for (int s = 0; s < 12; ++s)
		os << "#";

	os << "\n *** TRACKED DERIVATIVES *** : " << "\n";

	for (const auto& tracker : _derivatives)
		tracker.getValue().dump(os);

	for (int s = 0; s < 12; ++s)
		os << "#";

	os << "\n *** TRACKED REGEX(s) *** : " << "\n";

	for (const auto &regex : _regex) {
		os << "Regex: /" << regex.c_str() << "/";
	}
	 */
}

bool VariableFilter::isEnabled() const
{
	return _enabled;
}

void VariableFilter::setEnabled(bool enabled)
{
	this->_enabled = enabled;
}

void VariableFilter::addVariable(Tracker var)
{
	setEnabled(true);
	_variables[var.getName()] = var;
}

void VariableFilter::addDerivative(Tracker var)
{
	setEnabled(true);
	_derivatives[var.getName()] = var;
}

void VariableFilter::addRegexString(llvm::StringRef regex)
{
	setEnabled(true);
	_regex.push_back(regex.str());
}

VariableFilter::Filter VariableFilter::getVariableInfo(llvm::StringRef name, unsigned int expectedRank) const
{
	bool visibility = !isEnabled();
	llvm::SmallVector<ArrayRange, 3> ranges;

	if (matchesRegex(name))
	{
		visibility = true;
		ArrayRange unboundedRange(ArrayRange::unbounded, ArrayRange::unbounded);
		ranges.insert(ranges.begin(), expectedRank, unboundedRange);
	}

	if (_variables.count(name) != 0)
	{
		visibility = true;
		auto tracker = _variables.lookup(name);
		ranges.clear();

		// If the requested rank is lower than the one known by the variable filter,
		// then only keep an amount of ranges equal to the rank.

		auto trackerRanges = tracker.getRanges();
		unsigned int amount = expectedRank < trackerRanges.size() ? expectedRank : trackerRanges.size();
		auto it = trackerRanges.begin();
		ranges.insert(ranges.begin(), it, it + amount);
	}

	// If the requested rank is higher than the one known by the variable filter,
	// then set the remaining ranges as unbounded.

	for (size_t i = ranges.size(); i < expectedRank; ++i)
		ranges.emplace_back(ArrayRange::unbounded, ArrayRange::unbounded);

	return VariableFilter::Filter(visibility, ranges);
}

Filter VariableFilter::getVariableDerInfo(llvm::StringRef name, unsigned int expectedRank) const
{
	bool visibility = false;
	llvm::SmallVector<ArrayRange, 3> ranges;

	if (_derivatives.count(name) != 0)
	{
		visibility = true;
		auto tracker = _derivatives.lookup(name);
	}

	// For now, derivatives are always fully printed

	for (size_t i = ranges.size(); i < expectedRank; ++i)
		ranges.emplace_back(ArrayRange::unbounded, ArrayRange::unbounded);

	return VariableFilter::Filter(visibility, ranges);
}

bool VariableFilter::matchesRegex(llvm::StringRef identifier) const
{
	return llvm::any_of(_regex, [&identifier](const auto& regex) {
		llvm::Regex llvmRegex(regex);
		return llvmRegex.match(identifier);
	});
}

llvm::Expected<VariableFilter> VariableFilter::fromString(llvm::StringRef str)
{
	VariableFilter vf;
	Parser parser(vf, str);

	if (auto error = parser.run(); error)
		return std::move(error);

	return vf;
}
