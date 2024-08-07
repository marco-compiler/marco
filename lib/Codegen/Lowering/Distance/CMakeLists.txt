set(SOURCES
    DatabaseReader.cpp
    EditDistance.cpp
    SentenceDistanceCalculator.cpp
    WordDistanceCalculator.cpp)

marco_add_library(Distance
    ${SOURCES})

# We use a simple directory structure to store the wordnet database.
# It will be comprised of some .csv files generated by a python script,
# which will provide fast access to the wordnet database.
set(TARGET_DIR "${CMAKE_BINARY_DIR}/wordnet")

# As opposed to CMAKE_SOURCE_DIR, CMAKE_CURRENT_SOURCE_DIR is
# the directory of the current CMakeLists.txt file. In this case,
# it is the directory of this file, which is exactly where the
# Python script is located.
set(PYTHON_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/install-db.py")

# Custom target to create the directory and run the Python script
add_custom_target(
    run_python_script ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Running Python script"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TARGET_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$ENV{PYTHONPATH} python3 ${PYTHON_SCRIPT} ${TARGET_DIR} ${CMAKE_BINARY_DIR} ||
        ${CMAKE_COMMAND} -E echo "Python script failed. Did you install NLTK?"
    COMMENT "Creating directory and running Python script with input directory: ${TARGET_DIR}"
)

# Install command to move the wordnet directory to the installation path.
install(
    DIRECTORY "${TARGET_DIR}/"
    DESTINATION "share/marco/wordnet"
    COMPONENT wordnet
)
