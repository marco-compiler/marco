//
// Created by Ale on 21/06/2021.
//

#include <llvm/ADT/StringRef.h>
#include "marco/utils/VariableFilter.h"
#include "llvm/Support/Regex.h"


void marco::VariableFilter::dump() {

	for (int s = 0; s < 12; ++s) printf("#");
	cout << "\n *** TRACKED VARIABLES *** : " << endl;
	for (const std::pair<std::string, VariableTracker> vt  : _variables) {
		vt.second.dump(); //dump current variable tracker
	}

	for (int s = 0; s < 12; ++s) printf("#");
	cout << "\n *** TRACKED DERIVATIVES *** : " << endl;
	for (const std::pair<std::string, VariableTracker> vt  : _derivatives) {
		vt.second.dump(); //dump current variable tracker
	}

	for (int s = 0; s < 12; ++s) printf("#");
	cout << "\n *** TRACKED REGEX(s) *** : " << endl;
	for (const auto &regex : _regex) {
		printf("Regex: /%s/", regex.c_str());
	}

}

void marco::VariableFilter::addRegexString(string regex) {
	_bypass = false; //turnoff bypass by default
	_regex.push_back(regex);
}

void marco::VariableFilter::addVariable(VariableTracker var) {
	_bypass = false; //turnoff bypass by default
	_variables.insert_or_assign(var.getName(), var); //overwrite if key already exists
}

void marco::VariableFilter::addDerivative(VariableTracker var) {
	assert(var.getIsDerivative()); //add only derivatives
	_bypass = false; //turnoff bypass by default
	_derivatives.insert_or_assign(var.getName(), var);
}

bool marco::VariableFilter::isBypass() const { return _bypass; }

void marco::VariableFilter::setBypass(bool bypass) { _bypass = bypass; }

bool marco::VariableFilter::matchesRegex(const string &identifier) {

	llvm::StringRef testRef("[a-z]+");
	llvm::Regex testReg(testRef, llvm::Regex::NoFlags);
	bool t = testReg.match("!abcde99");

	for (const string &regexString : _regex) {
		llvm::StringRef regexRef(regexString);
		llvm::Regex llvmRegex(regexRef);
		if (llvmRegex.match(identifier)) {
			return true;
		}
	}
	return false;
}


bool checkIfPresent(unordered_map<std::string, VariableTracker> m, std::string key) {

	// Key is not present
	if (m.find(key) == m.end())
		return false;

	return true;

}

VariableTracker marco::VariableFilter::lookupByIdentifier(const string &identifier) {
	assert(checkIfPresent(_variables, identifier));
	return _variables.find(identifier)->second; //return variable tracker with key <identifier>
}

bool marco::VariableFilter::checkTrackedIdentifier(const string &identifier) {
	if (_bypass) return true;
	return checkIfPresent(_variables, identifier);
}


bool marco::VariableFilter::printDerivative(const string &derivedVariableIdentifier) {
	if (checkIfPresent(_derivatives, derivedVariableIdentifier)) return true;
	return false;
}
