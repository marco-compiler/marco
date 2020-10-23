#include <modelica/frontend/LexerStateMachine.hpp>

using namespace std;
using namespace llvm;
using namespace modelica;

using State = ModelicaStateMachine::State;

namespace modelica
{
	raw_ostream& operator<<(raw_ostream& stream, const Token& obj)
	{
		if (obj == Token::AlgorithmKeyword)
			stream << "algorithm";
		else if (obj == Token::AndKeyword)
			stream << "and";
		else if (obj == Token::AnnotationKeyword)
			stream << "annotation";
		else if (obj == Token::Assignment)
			stream << ":=";
		else if (obj == Token::Begin)
			stream << "begin";
		else if (obj == Token::BlockKeyword)
			stream << "block";
		else if (obj == Token::BreakKeyword)
			stream << "break";
		else if (obj == Token::ClassKeyword)
			stream << "class";
		else if (obj == Token::Colons)
			stream << ":";
		else if (obj == Token::Comma)
			stream << ",";
		else if (obj == Token::ConnectKeyword)
			stream << "connect";
		else if (obj == Token::ConnectorKeyword)
			stream << "connector";
		else if (obj == Token::ConstantKeyword)
			stream << "constant";
		else if (obj == Token::ConstraynedByKeyword)
			stream << "constrainedby";
		else if (obj == Token::DerKeyword)
			stream << "der";
		else if (obj == Token::Different)
			stream << "<>";
		else if (obj == Token::DiscreteKeyword)
			stream << "discrete";
		else if (obj == Token::Division)
			stream << "/";
		else if (obj == Token::Dot)
			stream << ".";
		else if (obj == Token::EachKeyword)
			stream << "each";
		else if (obj == Token::ElementWiseExponential)
			stream << "^";
		else if (obj == Token::ElementWiseMinus)
			stream << "-";
		else if (obj == Token::ElementWiseMultilpy)
			stream << "*";
		else if (obj == Token::ElementWiseSum)
			stream << "+";
		else if (obj == Token::ElementWiseDivision)
			stream << "/";
		else if (obj == Token::ElseIfKeyword)
			stream << "elseif";
		else if (obj == Token::ElseKeyword)
			stream << "else";
		else if (obj == Token::ElseWhenKeyword)
			stream << "elsewhen";
		else if (obj == Token::EncapsulatedKeyword)
			stream << "encapsulated";
		else if (obj == Token::End)
			stream << "EOF";
		else if (obj == Token::EndKeyword)
			stream << "end";
		else if (obj == Token::EnumerationKeyword)
			stream << "enumeration";
		else if (obj == Token::Equal)
			stream << "=";
		else if (obj == Token::EquationKeyword)
			stream << "equation";
		else if (obj == Token::Error)
			stream << "Error";
		else if (obj == Token::ExpandableKeyword)
			stream << "expandable";
		else if (obj == Token::Exponential)
			stream << "^";
		else if (obj == Token::ExtendsKeyword)
			stream << "extends";
		else if (obj == Token::ExternalKeyword)
			stream << "external";
		else if (obj == Token::FalseKeyword)
			stream << "false";
		else if (obj == Token::FinalKeyword)
			stream << "final";
		else if (obj == Token::FloatingPoint)
			stream << "FloatingPoint";
		else if (obj == Token::FlowKeyword)
			stream << "flow";
		else if (obj == Token::ForKeyword)
			stream << "for";
		else if (obj == Token::FunctionKeyword)
			stream << "function";
		else if (obj == Token::GreaterEqual)
			stream << ">=";
		else if (obj == Token::GreaterThan)
			stream << ">";
		else if (obj == Token::Ident)
			stream << "Identifier";
		else if (obj == Token::IfKeyword)
			stream << "if";
		else if (obj == Token::ImportKeyword)
			stream << "import";
		else if (obj == Token::ImpureKeyword)
			stream << "impure";
		else if (obj == Token::InitialKeyword)
			stream << "initial";
		else if (obj == Token::InKeyword)
			stream << "in";
		else if (obj == Token::InnerKeyword)
			stream << "inner";
		else if (obj == Token::InputKeyword)
			stream << "input";
		else if (obj == Token::Integer)
			stream << "Integer";
		else if (obj == Token::LCurly)
			stream << "{";
		else if (obj == Token::LPar)
			stream << "(";
		else if (obj == Token::LSquare)
			stream << "[";
		else if (obj == Token::LessEqual)
			stream << "<=";
		else if (obj == Token::LessThan)
			stream << "<";
		else if (obj == Token::LoopKeyword)
			stream << "loop";
		else if (obj == Token::Minus)
			stream << "-";
		else if (obj == Token::ModelKeyword)
			stream << "model";
		else if (obj == Token::Multiply)
			stream << "*";
		else if (obj == Token::None)
			stream << "None";
		else if (obj == Token::NotKeyword)
			stream << "not";
		else if (obj == Token::OperatorEqual)
			stream << "==";
		else if (obj == Token::OperatorKeyword)
			stream << "operator";
		else if (obj == Token::OrKeyword)
			stream << "or";
		else if (obj == Token::OuterKeyword)
			stream << "outer";
		else if (obj == Token::OutputKeyword)
			stream << "output";
		else if (obj == Token::PackageKeyword)
			stream << "package";
		else if (obj == Token::ParameterKeyword)
			stream << "parameter";
		else if (obj == Token::PartialKeyword)
			stream << "partial";
		else if (obj == Token::Plus)
			stream << "+";
		else if (obj == Token::ProtectedKeyword)
			stream << "protected";
		else if (obj == Token::PublicKeyword)
			stream << "public";
		else if (obj == Token::PureKeyword)
			stream << "pure";
		else if (obj == Token::RCurly)
			stream << "}";
		else if (obj == Token::RPar)
			stream << ")";
		else if (obj == Token::RSquare)
			stream << "]";
		else if (obj == Token::RecordKeyword)
			stream << "record";
		else if (obj == Token::RedeclareKeyword)
			stream << "redeclare";
		else if (obj == Token::ReplaceableKeyword)
			stream << "replaceable";
		else if (obj == Token::ReturnKeyword)
			stream << "return";
		else if (obj == Token::Semicolons)
			stream << ";";
		else if (obj == Token::StreamKeyword)
			stream << "stream";
		else if (obj == Token::String)
			stream << "String";
		else if (obj == Token::ThenKeyword)
			stream << "then";
		else if (obj == Token::TrueKeyword)
			stream << "true";
		else if (obj == Token::TypeKeyword)
			stream << "type";
		else if (obj == Token::WhenKeyword)
			stream << "when";
		else if (obj == Token::WhileKeyword)
			stream << "while";
		else if (obj == Token::WhithinKeyword)
			stream << "whithin";

		return stream;
	}
}

static bool isDigit(char c) { return ('0' <= c && c <= '9'); }

static bool isNonDigit(char c)
{
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_';
}

/**
 * Get the escaped version of a character, but ony if that escaped version has
 * a special meaning in the ASCII table.
 * For example, the escaped version of 'n' is a new line, but the escaped
 * version of 'z' means nothing special and thus only 'z' would be returned.
 */
static char escapedChar(char c)
{
	constexpr int alert = 7;
	constexpr int backspace = 8;
	constexpr int fromFeed = 12;

	switch (c)
	{
		case ('a'):
			return static_cast<char>(alert);
		case ('b'):
			return static_cast<char>(backspace);
		case ('f'):
			return static_cast<char>(fromFeed);
		case ('n'):
			return '\n';
		case ('r'):
			return '\r';
		case ('t'):
			return '\t';
		case ('v'):
			return '\v';
		default:
			return c;
	}
}

Token ModelicaStateMachine::charToToken(char c) const
{
	if (auto iter = symbols.find(c); iter != symbols.end())
		return iter->second;

	return Token::Error;
}

Token ModelicaStateMachine::stringToToken(const std::string& str) const
{
	if (auto iter = keywordMap.find(str); iter != keywordMap.end())
		return iter->getValue();

	return Token::Ident;
}

ModelicaStateMachine::ModelicaStateMachine(char first)
		: state(State::Normal),
			current('\0'),
			next(first),
			currentToken(Token::Begin),
			lastIdentifier(""),
			lastString(""),
			lineNumber(1),
			columnNumber(0)
{
	keywordMap["algorithm"] = Token::AlgorithmKeyword;
	keywordMap["and"] = Token::AndKeyword;
	keywordMap["annotation"] = Token::AnnotationKeyword;
	keywordMap["block"] = Token::BlockKeyword;
	keywordMap["break"] = Token::BreakKeyword;
	keywordMap["class"] = Token::ClassKeyword;
	keywordMap["connect"] = Token::ConnectKeyword;
	keywordMap["connector"] = Token::ConnectorKeyword;
	keywordMap["constant"] = Token::ConstantKeyword;
	keywordMap["constrainedby"] = Token::ConstraynedByKeyword;
	keywordMap["der"] = Token::DerKeyword;
	keywordMap["discrete"] = Token::DiscreteKeyword;
	keywordMap["each"] = Token::EachKeyword;
	keywordMap["else"] = Token::ElseKeyword;
	keywordMap["elseif"] = Token::ElseIfKeyword;
	keywordMap["elsewhen"] = Token::ElseWhenKeyword;
	keywordMap["encapsulated"] = Token::EncapsulatedKeyword;
	keywordMap["end"] = Token::EndKeyword;
	keywordMap["enumeration"] = Token::EnumerationKeyword;
	keywordMap["equation"] = Token::EquationKeyword;
	keywordMap["expandable"] = Token::ExpandableKeyword;
	keywordMap["extends"] = Token::ExtendsKeyword;
	keywordMap["external"] = Token::ExternalKeyword;
	keywordMap["false"] = Token::FalseKeyword;
	keywordMap["final"] = Token::FinalKeyword;
	keywordMap["flow"] = Token::FlowKeyword;
	keywordMap["for"] = Token::ForKeyword;
	keywordMap["function"] = Token::FunctionKeyword;
	keywordMap["if"] = Token::IfKeyword;
	keywordMap["import"] = Token::ImportKeyword;
	keywordMap["impure"] = Token::ImpureKeyword;
	keywordMap["in"] = Token::InKeyword;
	keywordMap["initial"] = Token::InitialKeyword;
	keywordMap["inner"] = Token::InnerKeyword;
	keywordMap["input"] = Token::InputKeyword;
	keywordMap["loop"] = Token::LoopKeyword;
	keywordMap["model"] = Token::ModelKeyword;
	keywordMap["not"] = Token::NotKeyword;
	keywordMap["operator"] = Token::OperatorKeyword;
	keywordMap["or"] = Token::OrKeyword;
	keywordMap["outer"] = Token::OuterKeyword;
	keywordMap["output"] = Token::OutputKeyword;
	keywordMap["package"] = Token::PackageKeyword;
	keywordMap["parameter"] = Token::ParameterKeyword;
	keywordMap["partial"] = Token::PartialKeyword;
	keywordMap["protected"] = Token::ProtectedKeyword;
	keywordMap["public"] = Token::PublicKeyword;
	keywordMap["pure"] = Token::PureKeyword;
	keywordMap["record"] = Token::RecordKeyword;
	keywordMap["redeclare"] = Token::RedeclareKeyword;
	keywordMap["replaceable"] = Token::ReplaceableKeyword;
	keywordMap["return"] = Token::ReturnKeyword;
	keywordMap["stream"] = Token::StreamKeyword;
	keywordMap["then"] = Token::ThenKeyword;
	keywordMap["true"] = Token::TrueKeyword;
	keywordMap["type"] = Token::TypeKeyword;
	keywordMap["when"] = Token::WhenKeyword;
	keywordMap["while"] = Token::WhileKeyword;
	keywordMap["whithin"] = Token::WhithinKeyword;

	symbols['*'] = Token::Multiply;
	symbols['-'] = Token::Minus;
	symbols['+'] = Token::Plus;
	symbols['/'] = Token::Division;
	symbols['.'] = Token::Dot;
	symbols['['] = Token::LSquare;
	symbols[']'] = Token::RSquare;
	symbols['{'] = Token::LCurly;
	symbols['}'] = Token::RCurly;
	symbols['('] = Token::LPar;
	symbols[')'] = Token::RPar;
	symbols['='] = Token::Equal;
	symbols['<'] = Token::LessThan;
	symbols['>'] = Token::GreaterThan;
	symbols[':'] = Token::Colons;
	symbols[';'] = Token::Semicolons;
	symbols[','] = Token::Comma;
	symbols['^'] = Token::Exponential;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingComment>()
{
	if (next == '\0')
		state = State::Normal;

	if (current == '*' && next == '/')
		state = State::EndOfComment;

	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::EndOfComment>()
{
	state = State::Normal;
	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingLineComment>()
{
	if (next == '\0')
		state = State::Normal;

	if (current == '\n')
		state = State::Normal;

	return Token::None;
}

static Token elementWise(char current, char next)
{
	if (current != '.')
		return Token::None;

	switch (next)
	{
		case ('/'):
			return Token::ElementWiseDivision;
		case ('*'):
			return Token::ElementWiseMultilpy;
		case ('-'):
			return Token::ElementWiseMinus;
		case ('+'):
			return Token::ElementWiseSum;
		case ('^'):
			return Token::ElementWiseExponential;
	}

	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingNum>()
{
	if (isDigit(current))
		lastNum.addUpper(current - '0');

	if (current == '.')
	{
		state = State::ParsingFloat;
		return Token::None;
	}

	if (current == 'E' || current == 'e')
	{
		state = State::ParsingFloatExponentialSign;
		return Token::None;
	}

	auto isAccetable = [](char c) {
		return isDigit(c) || c == '.' || c == 'e' || c == 'E';
	};

	if (!isAccetable(next))
	{
		state = State::Normal;
		return Token::Integer;
	}

	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingFloatExponent>()
{
	if (isDigit(current))
		lastNum.addExponential(current - '0');

	if (isDigit(next))
		return Token::None;

	state = State::Normal;
	return Token::FloatingPoint;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingFloatExponentialSign>()
{
	if (current == '-' || current == '+')
	{
		if (!isDigit(next))
		{
			error = "Exp sign must be followed by a number";
			state = State::Normal;
			return Token::Error;
		}

		state = State::ParsingFloatExponent;
		lastNum.setSign(current == '+');
		return Token::None;
	}

	if (isDigit(current))
	{
		state = State::ParsingFloatExponent;
		lastNum.addExponential(current - '0');
		return Token::None;
	}

	error = "Error unexpected char " + std::to_string(current) +
					" in floating number scan";
	state = State::Normal;
	return Token::Error;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingFloat>()
{
	if (isDigit(current))
		lastNum.addLower(current - '0');

	if (current == 'E' || current == 'e')
	{
		state = State::ParsingFloatExponentialSign;
		return Token::None;
	}

	if (isDigit(next) || next == 'E' || next == 'e')
		return Token::None;

	state = State::Normal;
	return Token::FloatingPoint;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingId>()
{
	lastIdentifier.push_back(current);

	if (!isDigit(next) && !isNonDigit(next))
	{
		state = State::Normal;
		return stringToToken(lastIdentifier);
	}

	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingString>()
{
	if (current == '"')
	{
		state = State::Normal;
		return Token::String;
	}

	if (current == '\\')
	{
		state = State::ParsingBackSlash;
		return Token::None;
	}

	if (current == '\0')
	{
		state = State::End;
		error = "Reached end of string while parsing a string";
		return Token::Error;
	}

	lastString.push_back(current);
	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingQId>()
{
	if (current == '\\')
	{
		state = State::ParsingIdBackSlash;
		return Token::None;
	}

	if (current == '\'')
	{
		state = State::Normal;
		return Token::Ident;
	}

	if (next == '\0')
	{
		state = State::Normal;
		error = "Unexpected end of string when parsing qidentifier";
		return Token::Error;
	}

	lastIdentifier.push_back(current);
	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingIdBackSlash>()
{
	lastIdentifier.push_back(escapedChar(current));
	state = State::ParsingQId;
	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingBackSlash>()
{
	lastString.push_back(escapedChar(current));
	state = State::ParsingString;
	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::Normal>()
{
	if (std::isspace(current) != 0)
		return Token::None;

	if (isNonDigit(current))
	{
		state = State::ParsingId;
		lastIdentifier = "";

		return scan<State::ParsingId>();
	}

	if (current == '\'')
	{
		state = State::ParsingQId;
		lastIdentifier = "";
		return Token::None;
	}

	if (isDigit(current))
	{
		state = State::ParsingNum;
		lastNum = FloatLexer<defaultBase>();
		return scan<State::ParsingNum>();
	}

	if (current == '/' && next == '/')
	{
		state = State::ParsingLineComment;
		return Token::None;
	}

	if (current == '/' && next == '*')
	{
		state = State::ParsingComment;
		return Token::None;
	}

	if (current == '"')
	{
		state = State::ParsingString;
		lastString = "";
		return Token::None;
	}

	if (current == '\0')
	{
		state = State::End;
		return Token::End;
	}

	return tryScanSymbol();
}

/**
 * Try to scan the next symbol by taking into account both the current and the
 * next characters. This avoids the need to define custom states to recognize
 * simple symbols such as '==' or ':='
 */
Token ModelicaStateMachine::tryScanSymbol()
{
	state = State::IgnoreNextChar;

	if (current == '<' && next == '>')
		return Token::Different;

	if (current == '<' && next == '=')
		return Token::LessEqual;

	if (current == '>' && next == '=')
		return Token::GreaterEqual;

	if (next == '=' && current == '=')
		return Token::OperatorEqual;

	if (current == ':' && next == '=')
		return Token::Assignment;

	if (Token token = elementWise(current, next); token != Token::None)
		return token;

	state = State::Normal;
	Token token = charToToken(current);

	if (token == Token::Error)
	{
		error = "Unexpected character ";
		error.push_back(current);
	}

	return token;
}

Token ModelicaStateMachine::step(char c)
{
	advance(c);

	switch (state)
	{
		case (State::Normal):
			return scan<State::Normal>();
		case (State::ParsingComment):
			return scan<State::ParsingComment>();
		case (State::ParsingLineComment):
			return scan<State::ParsingLineComment>();
		case (State::EndOfComment):
			return scan<State::EndOfComment>();
		case (State::ParsingNum):
			return scan<State::ParsingNum>();
		case (State::ParsingFloat):
			return scan<State::ParsingFloat>();
		case (State::ParsingFloatExponentialSign):
			return scan<State::ParsingFloatExponentialSign>();
		case (State::ParsingFloatExponent):
			return scan<State::ParsingFloatExponent>();
		case (State::ParsingString):
			return scan<State::ParsingString>();
		case (State::ParsingBackSlash):
			return scan<State::ParsingBackSlash>();
		case (State::ParsingId):
			return scan<State::ParsingId>();
		case (State::ParsingQId):
			return scan<State::ParsingQId>();
		case (State::ParsingIdBackSlash):
			return scan<State::ParsingIdBackSlash>();
		case (State::End):
			return Token::End;
		case (State::IgnoreNextChar):
			state = State::Normal;
			return Token::None;
	}

	error = "Unhandled Lexer State";
	return Token::Error;
}
