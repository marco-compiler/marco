//
// Created by Ale on 21/06/2021.
//

#include "../include/modelica/utils/VariableFilter.h"

void modelica::VariableFilter::dump() {

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

void modelica::VariableFilter::addRegexString(string regex) {
    _bypass = false; //turnoff bypass by default
    _regex.push_back(regex);
}

void modelica::VariableFilter::addVariable(VariableTracker var) {
    _bypass = false; //turnoff bypass by default
    _variables.insert_or_assign(var.getName(), var); //overwrite if key already exists
}

bool modelica::VariableFilter::isBypass() const { return _bypass; }

void modelica::VariableFilter::setBypass(bool bypass) { _bypass = bypass; }

bool modelica::VariableFilter::matchesRegex(const string &identifier) {
    for (const auto &regexString : _regex) {
        std::regex regex(regexString);
        if (regex_match(identifier, regex)) {
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

VariableTracker modelica::VariableFilter::lookupByIdentifier(const string &identifier) {
    assert(checkIfPresent(_variables, identifier));
    return _variables.find(identifier)->second; //return variable tracker with key <identifier>
}

bool modelica::VariableFilter::checkTrackedIdentifier(const string &identifier) {
    if (_bypass) return true;
    return checkIfPresent(_variables, identifier);
}


void modelica::VariableFilter::addDerivative(VariableTracker var) {
    assert(var.getIsDerivative()); //add only derivatives
    _derivatives.insert_or_assign(var.getName(), var);
}

bool modelica::VariableFilter::printDerivative(const string &derivedVariableIdentifier) {
    if(checkIfPresent(_derivatives, derivedVariableIdentifier)) return true;
    return false;
}
