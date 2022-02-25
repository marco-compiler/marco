set(SOURCES
		Utils.cpp
		Lexer.cpp
		Interval.cpp
		Range.cpp
		LogMessage.cpp
		SourcePosition.cpp
		VariableFilter.cpp)

if(MINGW)
set(LINK_MODE STATIC)
else()
set(LINK_MODE SHARED)
endif()
marco_add_library(utils ${LINK_MODE}
		${SOURCES}

		LINK_COMPONENTS
		Core
		Support)
