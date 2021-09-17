//
// Created by Ale on 26/06/2021.
//


#include "../include/modelica/utils/VariableFilterParser.h"

void modelica::VariableFilterParser::throwError(string msg) {
    cout << "\n\tVF Error: " << msg << endl;
    exit(1); //can't proceed with parsing
}

void modelica::VariableFilterParser::throwGenericError(void) {
    cout << "VF: ERROR when parsing | generic." << endl;
    exit(1);
}

bool modelica::VariableFilterParser::isValidIDChar(char c) {
    return (isalnum(c) || (c == '(') || (c == ')') || (c == '_'));
}

int modelica::VariableFilterParser::gettok() {
    int LastChar = 32;    //start from white space 32 === ' '
    rangeValue = ""; //empty the range value buffer
    // Skip any whitespace between tokens.
    while (isspace(LastChar)) {
        LastChar = (unsigned char) inputString[i];
        i++;
    }

    // VARIABLES AND DERIVATIVES (cannot start with a number)

    /**
     * The first form always start with a letter or underscore (_),
     * followed by any number of letters, digits, or underscores.
     */
    if (isalpha(LastChar) || (LastChar == '_')) {     // identifier: [a-zA-Z][a-zA-Z0-9]* or underscore

        IdentifierStr = LastChar;

        //single variable derivative der(x)
        if ((inputString[i]) == ')') {
            return tok_identifier;
        }
        if (inputString[i] == '[') { //starting to parse an array
            parsingArray = true;
            return tok_identifier;
        }

        if (LastChar == '_' && inputString[i] == '\0') {
            return LastChar;
        }
        else if (!isValidIDChar((inputString[i]))) {
            if(inputString[i] == '\0') {
                return tok_identifier;
            }
            else {
                return inputString[i]; //generic
            }
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
        } //while
        if(LastChar=='\0') return tok_identifier;

    }

    else if (LastChar == '/') {
        //keep reading
        while ((LastChar = inputString[i++]) != '/') {
            RegexStr += LastChar;
        }
        return tok_regex_expr;
    }

    else {
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
                if(parsingArray) break; //integer is array range value
                else return LastChar;

        }
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

void modelica::VariableFilterParser::lexerReset() {
    CurTok = 0;
    i = 0;
    parsingArray = false;
}

void modelica::VariableFilterParser::displayWarning(std::string msg) {
    cout << "\n\t(⚠️) VF Warning:️" << msg << endl;
}

void modelica::VariableFilterParser::parseCommandLine(string commandLineArguments, modelica::VariableFilter &vf) {
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
        return;
    }
}

void modelica::VariableFilterParser::parseExpressionElement(modelica::VariableFilter &vf) {

    string ele = inputStringReference;
    // cout << "\n\n*** NEW PARSING of " << ele << endl;
    inputString = ele.c_str();

    bool flag = true;
    lexerReset(); //reset the state (fresh start)
    getNextToken(); //start

    if (CurTok >= 0) {
        std::string base = "wrong input, unexpected token: ";
        throwError(base + ((char) CurTok));
        return;
    }
    while (flag) {
        //cout << "ready:";
        switch (CurTok) {
            case tok_eof: //end of input
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

                    unique_ptr<ArrayExprAST> arrayNode = ParseArray();

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

                unique_ptr<VariableExprAST> varExpr = ParseVariableExpr();

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

}
