include(GoogleTest)

ADD_EXECUTABLE(lowererTest src/LowererTest.cpp)
ADD_EXECUTABLE(modelica::lowererTest ALIAS lowererTest) 
TARGET_LINK_LIBRARIES(lowererTest PRIVATE gtest gtest_main modelica::lowerer)
TARGET_INCLUDE_DIRECTORIES(lowererTest PUBLIC include PRIVATE src)
TARGET_COMPILE_FEATURES(lowererTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      lowererTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)
