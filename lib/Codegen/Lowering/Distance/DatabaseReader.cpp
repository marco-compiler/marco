#include "marco/Codegen/Lowering/Distance/DatabaseReader.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace marco::codegen::lowering
{
    DatabaseReader& DatabaseReader::getInstance()
    {
        static DatabaseReader instance;
        return instance;
    }

    // As a constructor, this function opens the files if they haven't been opened yet.
    // Files are static, so they are shared among all instances of DatabaseReader. They
    // are not const, however, because reading from them changes their state.
    DatabaseReader::DatabaseReader()
    {
        filesOpened = false;
        std::string path = "../share/marco/wordnet";

        // Try to open the files using the shared path.
        senses.open(path + "/senses.csv");
        if (!senses.is_open())
        {
            // If something goes wrong, then we assume we're running in the
            // development environment, and we try to open the files by moving
            // up one directory.
            path = "../wordnet";
        
            senses.open(path + "/senses.csv");
            if (!senses.is_open())
            {
                return;
            }
        }

        hypernyms.open(path + "/hypernyms.csv");
        if (!hypernyms.is_open())
        {
            senses.close();
            return;
        }

        synsets.open(path + "/synsets.csv");
        if (!synsets.is_open())
        {
            senses.close();
            hypernyms.close();
            return;
        }

        std::ifstream synsetsCount(path + "/count.txt");
        if (!synsetsCount.is_open())
        {
            senses.close();
            hypernyms.close();
            synsets.close();
            return;
        }
        synsetsCount >> totalSynsetCount;
        filesOpened = true;
    }

    std::vector<std::string>
    DatabaseReader::keyToRow(std::ifstream& file, llvm::StringRef key)
    {
        if (!filesOpened) { return {}; }

        // The key is the first column in the CSV file, we exported them
        // in alphabetical order, so we can use binary search to find the
        // key.    
        int low = 0;
        int high = 0;
        file.seekg(0, std::ios::end);
        high = file.tellg();
        high--;

        // We keep track of the last explored key. This is to exit the loop
        // if we've converged to the same key twice in a row. This means we
        // haven't found the right row. The standard binary search algorithm
        // would just use the low and high pointers to find the key, but we
        // slightly modify those pointers to snap to the nearest newline,
        // so we need a stronger condition to exit the loop.
        std::string lastCurrentKey = "";
        while (low <= high)
        {
            int mid = low + (high - low) / 2;
            file.seekg(mid, std::ios::beg);

            // Keep moving the file pointer left until we find a newline.
            while (file.peek() != '\n' && mid > 0)
            {
                mid--;
                file.seekg(mid, std::ios::beg);
            }

            // Move the file pointer ahead of the newline.
            file.seekg(1, std::ios::cur);

            // Read the line.
            std::string line;
            std::getline(file, line);

            // Find the key.
            int firstComma = line.find(',');
            std::string currentKey = line.substr(0, firstComma);

            // If the current key is the same as the last key, we
            // haven't found the right row. Break out of the loop.
            if (currentKey == lastCurrentKey)
            {
                break;
            }

            lastCurrentKey = currentKey;

            // Compare the alphabetically ordered keys.
            if (currentKey < key) { low = mid + 1; }
            else if (currentKey > key) { high = mid - 1; }
            else
            {
                // Remove the first column (the key).
                line = line.substr(line.find(',') + 1);

                // Split the line into columns.
                std::vector<std::string> columns;
                std::string column;
                for (char c : line)
                {
                    if (c == ',')
                    {
                        columns.push_back(column);
                        column.clear();
                    }
                    else { column += c; }
                }

                // Add the last column.
                columns.push_back(column);
                return columns;
            }
        }

        // If the key wasn't found, return an empty vector.
        return {};
    }

    std::vector<Synset>
    DatabaseReader::getSynsets(llvm::StringRef word)
    {
        if (!filesOpened) { return {}; }

        std::vector<std::string> columns = keyToRow(senses, word);
        std::vector<Synset> synsets;

        for (const std::string& column : columns)
        {
            synsets.push_back(std::stoi(column));
        }

        return synsets;
    }

    std::vector<std::string>
    DatabaseReader::offsetToRow(std::ifstream& file, int offset)
    {
        if (offset < 0 || !filesOpened) { return {}; }

        file.clear();
        file.seekg(offset, std::ios::beg);

        std::string line;
        std::getline(file, line);


        std::vector<std::string> columns;
        std::string column;
        for (char c : line)
        {
            if (c == ',')
            {
                columns.push_back(column);
                column.clear();
            }
            else
            {
                column += c;
            }
        }

        columns.push_back(column);

        // Remove the first column (the key).
        columns.erase(columns.begin());

        return columns;
    }

    std::vector<Synset>
    DatabaseReader::getHypernyms(const Synset& synset)
    {
        if (synset == -1 || !filesOpened) { return {}; }

        std::vector<std::string> columns = offsetToRow(hypernyms, synset);
        std::vector<Synset> hypernyms;

        // Remove the first column (the count).
        columns.erase(columns.begin());

        for (const std::string& column : columns)
        {
            hypernyms.push_back(std::stoi(column));
        }

        return hypernyms;
    }

    std::string DatabaseReader::getSynsetName(const Synset& synset)
    {
        // If the synset is invalid, return an empty string.
        if (synset < 0 || !filesOpened) { return ""; }

        // Turn the synset into a 8-character-long string.
        std::string synsetStr = std::to_string(synset);
        while (synsetStr.size() < 8)
        {
            synsetStr = "0" + synsetStr;
        }

        // Find the synset in the synsets file.
        std::vector<std::string> columns = keyToRow(synsets, synsetStr);

        // Return the first column (the name).
        return columns[0];
    }

    int DatabaseReader::getSynsetCount(const Synset& synset)
    {
        if (synset == -1 || !filesOpened) { return 0; }

        std::vector<std::string> columns = offsetToRow(hypernyms, synset);
        return std::stoi(columns[0]);
    }

    int DatabaseReader::getTotalSynsetCount() const
    {
        return totalSynsetCount;
    }
}