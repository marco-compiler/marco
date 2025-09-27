set(SOURCES
        Location.cpp)

marco_add_library(Lexer
        ${SOURCES}

        LINK_COMPONENTS
        Core
        Support)
