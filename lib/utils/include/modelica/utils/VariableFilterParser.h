//
// Created by Ale on 26/06/2021.
//

#ifndef PARSER_LEXER_M_VF_VARIABLEFILTERPARSER_H
#define PARSER_LEXER_M_VF_VARIABLEFILTERPARSER_H

#include <regex>
#include <iostream>
#include <cstring>
#include <list>
#include "VariableFilter.h"

using namespace std;
namespace modelica {

    class VariableFilterParser {

        static void throwGenericError(void) {
            cout << "VF: ERROR when parsing | generic." << endl;
            exit(0);
        }

        //	LEXER 	//
        //============================================================================================================//
        // The lexer returns tokens [0-255] if it is an unknown character, otherwise
        // one
        // of these for known things.
        enum Token {
            tok_eof = -1,

            // commands
            tok_der = -2,
            tok_der_close = -3,

            // primary
            tok_identifier = -4,

            //array support
            tok_sq_bracket_l = -5,
            tok_sq_bracket_r = -6,
            tok_comma = -7,
            tok_colon = -8,
            tok_rng_value = -9,

            //regex support
            tok_regex_expr


        };

        std::string IdentifierStr;     // Filled in if tok_identifier
        std::string RegexStr; //filled in if tok_regex_exp
        const char *inputString;
        unsigned short i = 0;
        bool parsingArray;
        string rangeValue;

        bool isValidIDChar(char c) {
            return (isalnum(c) || (c == '(') || (c == ')') || (c == '_'));
        }

        /// gettok - Return the next token from the input string. T O K E N I Z E R
        int gettok() {
            int LastChar = 32;    //start from white space 32 === ' '
            rangeValue = ""; //empty the range value buffer
            // Skip any whitespace between tokens.
            while (isspace(LastChar)) {
                LastChar = (unsigned char) inputString[i];
                i++;
            }

            // VARIABLES AND DERIVATIVES (cannot start with a number)
            if (isalpha(LastChar) || (LastChar == '_')) {     // identifier: [a-zA-Z][a-zA-Z0-9]* or underscore

                IdentifierStr = LastChar;

                //single variable derivative der(x)
                if ((inputString[i]) == ')') {
                    return tok_identifier;
                }
                if (inputString[i] == '[') {
                    parsingArray = true;
                    return tok_identifier;
                }
                //identifier: [a-zA-Z][a-zA-Z0-9]* or underscore
                while (isValidIDChar((LastChar = inputString[i++]))) {    // keep reading until identifier stops
                    IdentifierStr += LastChar;

                    if (IdentifierStr == "der(")
                        return tok_der;

                    //check next character
                    if ((inputString[i]) == ')') {
                        return tok_identifier;
                    }
                    if ((inputString[i]) == '[') {
                        parsingArray = true;
                        return tok_identifier;
                    }
                    if (IdentifierStr == ")") {
                        return tok_der_close;
                    }
                }

                return tok_identifier;
            }

            if (LastChar == '/') {
                //keep reading
                while ((LastChar = inputString[i++]) != '/') {
                    RegexStr += LastChar;
                }
				/* TODO: add regex check without exceptions
                try {
                    std::regex checkRegex(RegexStr, std::regex::ECMAScript);
                    return tok_regex_expr;
                }
                catch (regex_error e) {
                    throwGenericError();
                } */
                return tok_regex_expr;
            }

            switch (LastChar) {
                case ('['):
                    return tok_sq_bracket_l;
                case (']'):
                    return tok_sq_bracket_r;
                case (','):
                    return tok_comma;
                case (':'):
                    return tok_colon;
                case ('$'):
                    rangeValue += (char) LastChar;
                    return tok_rng_value;
                default:
                    break;

            }

            //range value of an array
            if (isdigit(LastChar)) {
                rangeValue += (char) LastChar;
                while (isdigit(LastChar = inputString[i])) {
                    rangeValue += (char) LastChar;
                    i++;
                }
                return tok_rng_value;
            }


            return LastChar;
        }

        // === AST / PARSER === //
        //============================================================================================================//

        /// ExprAST - Base class for all expression nodes.
        class ExprAST {
        public:
            virtual ~ExprAST() {}
        };

        class VariableExprAST : public ExprAST {
            string identifier;

        public:
            VariableExprAST(const string &identifier) : identifier(identifier) {}

            string getIdentifier() const { return identifier; }
        };

        class ArrayRangeAST : public ExprAST {
            const int lvalue, rvalue;

        public:
            ArrayRangeAST(const int lvalue, const int rvalue) : lvalue(lvalue), rvalue(rvalue) {}

             int getLvalue() const {
                return lvalue;
            }

             int getRvalue() const {
                return rvalue;
            }


        };

        class ArrayExprAST : public ExprAST {

        public:

            ArrayExprAST(const VariableExprAST &arrayVariableIdentifier, uint16_t dimension,
                         list<ArrayRangeAST> &ranges)
                    : arrayVariableIdentifier(arrayVariableIdentifier), dimension(dimension), _ranges(ranges) {}

            const VariableExprAST getArrayVariableIdentifier() const {
                return arrayVariableIdentifier;
            }

            uint16_t getDimension() const {
                return dimension;
            }

            list<ArrayRangeAST> getRanges() const {
                return _ranges;
            }


        private:

            VariableExprAST arrayVariableIdentifier;
            uint16_t dimension;
            list<ArrayRangeAST> _ranges;
        };

        class DerivativeExprAST : public ExprAST {
            VariableExprAST derivedVariable;


        public:
            explicit DerivativeExprAST(VariableExprAST derivedVariable)
                    : derivedVariable(std::move(derivedVariable)) {
            }

            const VariableExprAST &getDerivedVariable() const {
                return derivedVariable;
            }
        };

        class RegexExprAST : public ExprAST {
            string regex;

        public:
            explicit RegexExprAST(string regex) : regex(std::move(regex)) {}

            const string &getRegex() const {
                return regex;
            }
        };

        /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the
        /// current
        /// token the parser is looking at.  getNextToken reads another token from
        /// the lexer and updates CurTok with its results.
        int CurTok;

        int getNextToken() { return CurTok = gettok(); }

        //===========================================GRAMMAR PRODUCTIONS=================================================//
        //===============================================================================================================//
        /// var ::= string aAzZ09_
        std::unique_ptr<VariableExprAST> ParseVariableExpr() {
            std::string idName = IdentifierStr;
            auto result = std::make_unique<VariableExprAST>(idName);
            // getNextToken();	 // consume the number
            return std::move(result);
        }

        std::unique_ptr<DerivativeExprAST> ParseDerivative() {
            getNextToken();     // eat "der("

            // create derivative
            std::string IdName = IdentifierStr;
            auto Result = std::make_unique<DerivativeExprAST>(IdName);
            getNextToken();     // consume )
            return std::move(Result);

        }

        std::unique_ptr<ArrayExprAST> ParseArray() {

            //indices begin
            if (getNextToken() != tok_sq_bracket_l) throwGenericError(); //[
            //cout << "\n opening bracket [";
            std::string arrayName = IdentifierStr;
            uint16_t dimension = 0;

            std::list<ArrayRangeAST> local_ranges;

            int rtemp, ltemp;
            do {
                //cout << "\n\tnew range:: ";
                if (getNextToken() != tok_rng_value) throwGenericError();           //left-value
                // cout << "lval=" << rangeValue;
                ltemp = 0 == strncmp(rangeValue.c_str(), "$", 1) ? -1 : atoi(rangeValue.c_str());

                if (getNextToken() != tok_colon) throwGenericError();      // :
                // cout << " : ";
                if (getNextToken() != tok_rng_value) throwGenericError();    //right value
                // cout << "r-val=" << rangeValue;
                rtemp = 0 == strncmp(rangeValue.c_str(), "$", 1) ? -1 : atoi(rangeValue.c_str());

                ArrayRangeAST rang3(ltemp, rtemp);
                local_ranges.push_back(rang3);
                dimension++;

            } while (getNextToken() == tok_comma);

            //indices end
            if (CurTok != tok_sq_bracket_r) throwGenericError();
            //cout << " \nclosing bracket ]" << endl;

            auto result = std::make_unique<ArrayExprAST>(arrayName, dimension, local_ranges);
            //list<ArrayRangeAST> rntest = result->getRanges(); OK
            return result;

        }

        std::unique_ptr<RegexExprAST> ParseRegex() {

            std::string regex = RegexStr;
            auto result = std::make_unique<RegexExprAST>(regex);
            return std::move(result);
        }


    private:

        string inputStringReference;

        void lexerReset() {
            CurTok = 0;
            i = 0;
            parsingArray = false;
        }

        void displayWarning(std::string msg) {
            cout << "\n\t(⚠️) VF Warning:️" << msg << endl;
        }


    public:
        explicit VariableFilterParser(string commandLineString)
                : inputStringReference(std::move(commandLineString)) {
        }
        explicit VariableFilterParser() {}

        void parseCommandLine(string commandLineArguments, VariableFilter &vf) {
            size_t pos = 0;
            std::string delimiter = ";";
            std::string token;
            bool atLeastOne = false;
            while ((pos = commandLineArguments.find(delimiter)) != std::string::npos) {
                token = commandLineArguments.substr(0, pos);
                //std::cout << token << std::endl;
                inputStringReference = token;
                parseExpressionElement(vf); //parse each token
                commandLineArguments.erase(0, pos + delimiter.length());
                atLeastOne = true;
            }
            if(!atLeastOne) {
                displayWarning("No VF input provided.");
            }
        }
        /** parses the current input string and carries out parsing into var-regex-array-derivative by
         * adding a tracker in the VariableFilter vf*/
        void parseExpressionElement(VariableFilter &vf) {
            //SmallVector<StringRef> elements;
            //cout << "Splitting: " << inputStringReference.str() << endl;
            //inputStringReference.split(elements, ',', -1, false);

            string ele = inputStringReference;
            // cout << "\n\n*** NEW PARSING of " << ele << endl;
            inputString = ele.c_str();

            bool flag = true;
            lexerReset();
            while (flag) {
                //cout << "ready:";
                switch (CurTok) {
                    case tok_eof:
                        return;
                    case tok_der: {
                        unique_ptr<DerivativeExprAST> derivativeNode = ParseDerivative();
                        // cout << "\nDERIVATIVE DONE" << endl;
                        VariableTracker tracker(derivativeNode->getDerivedVariable().getIdentifier(), false, true,
                                                0);

                        //vf.addVariable(tracker);
                        //new derivative map
                        vf.addDerivative(tracker);
                        return;
                    }
                    case tok_identifier: {

                        if (parsingArray) {
                            //cout << "\n 🏁 parsing an array" << endl;
                            unique_ptr<ArrayExprAST> arrayNode = ParseArray();
                            //cout << "\nARRAY DONE" << endl;

                            string id = arrayNode->getArrayVariableIdentifier().getIdentifier();
                            VariableTracker tracker(id, true, false, arrayNode->getDimension());
                            list<ArrayRangeAST> rangeNodes = arrayNode->getRanges();
                            list<Range> rangeList;
                            for (auto r : rangeNodes) {
                                rangeList.emplace_back(Range(r.getLvalue(), r.getRvalue()));
                            }

                            tracker.setRanges(rangeList);
                            vf.addVariable(tracker);
                            return;
                        }

                        //cout << "parsing a variable identifier: ";
                        unique_ptr<VariableExprAST> varExpr = ParseVariableExpr();
                        // cout << "\nVAR ID DONE" << endl;

                        VariableTracker tracker(varExpr->getIdentifier(), false, false, 0);

                        vf.addVariable(tracker);
                        flag = false;
                        return;
                        //variable identifier parsed correctly
                    }

                    case tok_regex_expr : {
                        unique_ptr<RegexExprAST> regNode = ParseRegex();
                        vf.addRegexString(regNode->getRegex());
                        return;
                    }


                    default:
                        getNextToken();
                        break;
                }
            }

        };
    };

}


#endif //PARSER_LEXER_M_VF_VARIABLEFILTERPARSER_H
