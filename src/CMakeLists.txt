cmake_minimum_required(VERSION 3.10)

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

project(egs CXX)
# TODO add way to push version number into source code

find_package(Boost COMPONENTS system filesystem thread REQUIRED)
find_package(Threads REQUIRED)

#add_definitions( -DAPP_VERSION=\"0.1.0\" -DAPPLICATION_NAME=\"SiQAD\" -DORGANIZATION_NAME=\"WalusLab\" )

include_directories(
    .
    ${Boost_INCLUDE_DIR}
)


if(DEFINED SIQAD_PLUGINS_ROOT)
    set(EGS_INSTALL_DIR "${SIQAD_PLUGINS_ROOT}/exhaustive_gs")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(EGS_INSTALL_DIR "debug")
else()
    set(EGS_INSTALL_DIR "release")
ENDIF()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCES
    main.cc
    interface.cc
    exhaustive_gs.cc
    siqadconn.cc
)


add_executable(egs ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

#add_custom_command(
#    TARGET egs POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy
#            ${CMAKE_CURRENT_SOURCE_DIR}/simanneal.physeng
#            ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS egs RUNTIME DESTINATION ${EGS_INSTALL_DIR})
#install(FILES simanneal.physeng DESTINATION ${EGS_INSTALL_DIR})

